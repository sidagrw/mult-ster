import torch
from tqdm import tqdm
import numpy as np


def generate(model, tokenizer, prompt, device, max_new_tokens=512):
    """
    standard direct inference, 0-shot
    """

    model_input = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        len_prompt = model_input['input_ids'].shape[1]
        # check whether model is an instance of HookedTransformer
        if hasattr(model, 'W_in'):
            # token 32007 is phi-specific
            output = model.generate(model_input['input_ids'], max_new_tokens=max_new_tokens, do_sample=False, verbose=False, stop_at_eos=True, eos_token_id=[tokenizer.eos_token_id, 32007])[0,len_prompt:]
        else:
            output = model.generate(model_input['input_ids'], max_new_tokens=max_new_tokens, do_sample=False)[0,len_prompt:]
        
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
    
    return decoded_output


def adjust_vectors(v, u, target_values):
    """
    Adjusts a batch of vectors v such that their projections along the unit vector u equal the target values.

    Parameters:
    - v: A 2D tensor of shape (n, d), representing the batch of vectors to be adjusted.
    - u: A 1D unit tensor of shape (d,), representing the direction along which the adjustment is made.
    - target_values: A 1D tensor of shape (n,), representing the desired projection values of the vectors in v along u.

    Returns:
    - adjusted_v: The adjusted batch of vectors such that their projections along u are equal to the target values.
    """
    current_projections = v @ u  # Current projections of v onto u
    delta = target_values - current_projections  # Differences needed to reach the target projections
    adjusted_v = v + delta[:, None] * u  # Adjust v by the deltas along the direction of u
    return adjusted_v


def activation_addition_hook(
    activation,
    hook,
    direction,
    weight=1,
):
    return activation + (direction * weight)


def direction_projection_hook(
    activation,
    hook,
    direction,
    value_along_direction,
):
    adjusted_activations = adjust_vectors(activation.squeeze(), direction, value_along_direction)
    return adjusted_activations.unsqueeze(0)

def compute_task_matrix(X, X_plus):
    """
    X: (N, d) matrix of uninstructed activations
    X_plus: (N, d) matrix of instructed activations
    Returns: W (d, d) task matrix
    """
    # Solve X @ W.T = X_plus
    # W.T = pinv(X) @ X_plus
    W_T, _, _, _ = np.linalg.lstsq(X, X_plus, rcond=None)
    return W_T.T # Return W as (d, d)

def multiplicative_steering_hook(
    activation,  # Shape: [batch, seq_len, d]
    hook,
    task_matrix, # Shape: [d, d]
    alpha=1.0,
):
    # Calculate Wx'
    # activation.squeeze() handles the [1, 1, d] shape from HookedTransformer
    steered_act = torch.matmul(activation, task_matrix.t())
    
    # Apply interpolation formula: x' + alpha * (Wx' - x')
    new_activation = activation + alpha * (steered_act - activation)
    
    return new_activation


def generate_with_hooks(
    model,
    toks,
    max_tokens_generated: int = 64,
    fwd_hooks = [],
    verbose: bool = False,
    return_decoded=True
):
    """
    FAST VERSION: Uses KV-Caching and removes hallucinated 'Q:' yapping.
    """
    with torch.no_grad():
        # 1. Use the fast engine (KV-Caching is active here)
        # It handles the Multiplicative/Additive hooks automatically via fwd_hooks
        all_toks = model.generate(
            input=toks,
            max_new_tokens=max_tokens_generated,
            fwd_hooks=fwd_hooks,
            do_sample=False,
            verbose=verbose,
            stop_at_eos=True
        )

    # 2. Extract only the generated tokens
    generated_toks = all_toks[:, toks.shape[1]:]

    if return_decoded:
        # 3. Decode the tokens into text
        decoded_list = model.tokenizer.batch_decode(generated_toks, skip_special_tokens=True)
        
        # 4. THE "Q:" FIX (Post-Processing)
        # Instead of stopping token-by-token, we just cut the string 
        # if the model started hallucinating a new Question.
        cleaned_decoded = []
        for text in decoded_list:
            # Gemma often generates "\nQ:" or "Q:" when it starts yapping
            if "Q:" in text:
                text = text.split("Q:")[0]
            cleaned_decoded.append(text.strip())
            
        return cleaned_decoded
    else:
        return all_toks

def compute_perplexity(text, device='cuda', perplexity_model=None, perplexity_tokenizer=None):
    # Tokenize the input text
    inputs = perplexity_tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    # if longer than 1024 tokens, take the last 1024 tokens
    if input_ids.shape[1] > 1024:
        input_ids = input_ids[:, -1024:]

    # Compute the log probabilities
    with torch.no_grad():
        try:
            outputs = perplexity_model(input_ids, labels=input_ids)
            loss = outputs.loss  # This is the average negative log-likelihood per token
        except Exception as e:
            print(f'Error in computing perplexity for text: {text}')
            print(f'Error: {e}')
            loss = torch.tensor(0.0)

    # Compute the perplexity
    perplexity = torch.exp(loss)
    return perplexity.item()


def extract_representation(model, tokenizer, problem, device, num_final_tokens=8):
    """
    extract the representation of the final token in the direct inference prompt
    """
    eval_prompt = problem

    model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits, cache = model.run_with_cache(model_input['input_ids'])
        del logits
        final_token_rs = torch.stack([cache['resid_post', layer_idx][:, -num_final_tokens:, :].squeeze() for layer_idx in range(model.cfg.n_layers)]).cpu().numpy()
        del cache
    
    return final_token_rs