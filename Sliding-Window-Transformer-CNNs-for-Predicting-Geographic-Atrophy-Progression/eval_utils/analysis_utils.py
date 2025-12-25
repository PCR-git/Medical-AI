import numpy as np
from pathlib import Path
import os
from tabulate import tabulate

from scipy import stats
from typing import List, Tuple, Dict, Any


# def analyze_kfold_results(
#     base_dir,
#     RESIDUAL_HISTORY_FILE,
#     MASK_HISTORY_FILE,
#     MEAN_SCORE_KEY,
#     STD_SCORE_KEY,
#     n_epochs,
#     t_index,
#     last_epoch=None,      # NEW
#     fold_index=None       # NEW
# ):
#     """
#     Analyze k-fold results.

#     NEW:
#     - last_epoch: choose an epoch cutoff (None = end)
#     - fold_index: choose a single fold to analyze (None = all folds)
#     """
#     base_path = Path(base_dir)
#     print(f"Attempting to access base path: {base_path}")
#     if not base_path.exists():
#         print(f"CRITICAL ERROR: Python cannot verify the existence of the path: {base_path}")
#         return
    
#     print("Base path exists. Attempting to list contents...")
    
#     try:
#         fold_paths = sorted([p for p in base_path.iterdir() if p.is_dir()])
#     except Exception as e:
#         print(f"CRITICAL ERROR: Could not list directory. Error: {e}")
#         return

#     if not fold_paths:
#         print(f"Error: No subdirectories found in {base_path}.")
#         return

#     print(f"Found {len(fold_paths)} fold directories.")

#     # ---------------------------------------------------------------------
#     # NEW: restrict to a single fold if fold_index is provided
#     # ---------------------------------------------------------------------
#     if fold_index is not None:
#         if fold_index < 0 or fold_index >= len(fold_paths):
#             print(f"Error: fold_index={fold_index} out of range (0 to {len(fold_paths)-1})")
#             return
        
#         print(f"Selecting ONLY fold index {fold_index}: {fold_paths[fold_index].name}")
#         fold_paths = [fold_paths[fold_index]]
#     # ---------------------------------------------------------------------

#     residual_fold_means = []
#     residual_fold_stds = []
#     mask_fold_means = []
#     mask_fold_stds = []
#     processed_folds = 0

#     for fold_path in fold_paths:
#         print(f"-> Processing {fold_path.name}...")

#         def extract_dice_array(file_path, key):
#             history_dict = np.load(file_path, allow_pickle=True).item()
#             arr = np.asarray(history_dict[key])
#             if arr.ndim != 1:
#                 arr = arr[:, t_index]
#             return arr
        
#         current_fold_successful = False

#         # --- Residual History ---
#         res_file = fold_path / RESIDUAL_HISTORY_FILE
#         if res_file.exists():
#             try:
#                 res_score = extract_dice_array(res_file, MEAN_SCORE_KEY)
#                 res_std   = extract_dice_array(res_file, STD_SCORE_KEY)

#                 L = len(res_score)
#                 effective_last = L if last_epoch is None else last_epoch
#                 if effective_last > L:
#                     raise ValueError(f"last_epoch={last_epoch} exceeds history length {L}")

#                 start = max(0, effective_last - n_epochs)
#                 end   = effective_last

#                 residual_fold_means.append(np.mean(res_score[start:end]))
#                 residual_fold_stds.append(np.mean(res_std[start:end]))

#                 current_fold_successful = True

#             except Exception as e:
#                 print(f"   Error loading Residual: {e}")
#         else:
#             print(f"   Warning: Missing residual file: {res_file.name}")

#         # --- Mask History ---
#         mask_file = fold_path / MASK_HISTORY_FILE
#         if current_fold_successful and mask_file.exists():
#             try:
#                 mask_score = extract_dice_array(mask_file, MEAN_SCORE_KEY)
#                 mask_std   = extract_dice_array(mask_file, STD_SCORE_KEY)

#                 L = len(mask_score)
#                 effective_last = L if last_epoch is None else last_epoch
#                 if effective_last > L:
#                     raise ValueError(f"last_epoch={last_epoch} exceeds history length {L}")

#                 start = max(0, effective_last - n_epochs)
#                 end   = effective_last

#                 mask_fold_means.append(np.mean(mask_score[start:end]))
#                 mask_fold_stds.append(np.mean(mask_std[start:end]))

#                 processed_folds += 1

#             except Exception as e:
#                 print(f"   Error loading Mask: {e}. Skipping fold.")
#                 residual_fold_means.pop()
#                 residual_fold_stds.pop()

#         elif current_fold_successful:
#             print(f"   Warning: Missing mask file: {mask_file.name}. Skipping fold.")
#             residual_fold_means.pop()
#             residual_fold_stds.pop()

#     # ---------------------------------------------------------------------
#     # If a single fold was picked, return its results directly
#     # ---------------------------------------------------------------------
#     if fold_index is not None:
#         if processed_folds == 0:
#             print("\nNo valid data in requested fold.")
#             return
        
#         print("\n=== SINGLE-FOLD RESULT ===\n")
#         print(f"Fold: {fold_paths[0].name}")
#         print(f"Epochs averaged: {start}–{end}")

#         print(f"Residual mean: {residual_fold_means[0]:.4f}")
#         print(f"Residual std:  {residual_fold_stds[0]:.4f}")
#         print(f"Mask mean:     {mask_fold_means[0]:.4f}")
#         print(f"Mask std:      {mask_fold_stds[0]:.4f}")
#         return
#     # ---------------------------------------------------------------------

#     # --- Multi-fold summary ---
#     if processed_folds == 0:
#         print("\nAnalysis failed: zero folds processed.")
#         return

#     final_res_mean  = np.mean(residual_fold_means)
#     final_res_std   = np.std(residual_fold_means)
#     final_mask_mean = np.mean(mask_fold_means)
#     final_mask_std  = np.std(mask_fold_means)

#     results_data = [
#         ["Residual Mask (T=3)", f"{final_res_mean:.4f}", f"{final_res_std:.4f}", processed_folds],
#         ["Full Mask (T=3)",     f"{final_mask_mean:.4f}", f"{final_mask_std:.4f}", processed_folds],
#     ]

#     print("\n" + "="*50)
#     print(f"| FINAL K-FOLD CROSS-VALIDATION RESULTS ({processed_folds} FOLDS) |")
#     print("="*50 + "\n")
#     print(f"Scores averaged over epochs {start} to {end}.")

#     print(tabulate(
#         results_data,
#         headers=["Channel", "Mean Dice (μ)", "Std Dev (σ)", "Folds Used"],
#         tablefmt="fancy_grid"
#     ))


def analyze_kfold_results(
    base_dir,
    RESIDUAL_HISTORY_FILE,
    MASK_HISTORY_FILE,
    MEAN_SCORE_KEY,
    STD_SCORE_KEY,
    n_epochs,
    t_index,
    last_epoch=None,
    fold_index=None
):
    """
    Analyze k-fold results. Calculates the mean score and standard deviation 
    across the last n_epochs for each fold and returns the raw fold means for the t-test.
    """
    base_path = Path(base_dir)
    print(f"Attempting to access base path: {base_path}")
    if not base_path.exists():
        print(f"CRITICAL ERROR: Python cannot verify the existence of the path: {base_path}")
        # MUST return None to indicate failure before loading
        return None
    
    print("Base path exists. Attempting to list contents...")
    
    try:
        fold_paths = sorted([p for p in base_path.iterdir() if p.is_dir()])
    except Exception as e:
        print(f"CRITICAL ERROR: Could not list directory. Error: {e}")
        return None

    if not fold_paths:
        print(f"Error: No subdirectories found in {base_path}.")
        return None

    print(f"Found {len(fold_paths)} fold directories.")

    # ---------------------------------------------------------------------
    # NEW: restrict to a single fold if fold_index is provided
    # ---------------------------------------------------------------------
    if fold_index is not None:
        if fold_index < 0 or fold_index >= len(fold_paths):
            print(f"Error: fold_index={fold_index} out of range (0 to {len(fold_paths)-1})")
            return None
        
        print(f"Selecting ONLY fold index {fold_index}: {fold_paths[fold_index].name}")
        fold_paths = [fold_paths[fold_index]]
    # ---------------------------------------------------------------------

    residual_fold_means = []
    residual_fold_stds = []
    mask_fold_means = []
    mask_fold_stds = []
    processed_folds = 0
    start = 0
    end = 0

    for fold_path in fold_paths:
        print(f"-> Processing {fold_path.name}...")

        def extract_dice_array(file_path, key):
            # Using np.load without global dependency check for brevity
            history_dict = np.load(file_path, allow_pickle=True).item()
            arr = np.asarray(history_dict[key])
            if arr.ndim != 1:
                # Assumes T_CRITICAL_INDEX is 2 for T=3 prediction time step
                arr = arr[:, t_index]
            return arr
        
        current_fold_successful = False

        # --- Residual History (Growth Mask) ---
        res_file = fold_path / RESIDUAL_HISTORY_FILE
        if res_file.exists():
            try:
                res_score = extract_dice_array(res_file, MEAN_SCORE_KEY)
                res_std   = extract_dice_array(res_file, STD_SCORE_KEY)

                L = len(res_score)
                effective_last = L if last_epoch is None else last_epoch
                if effective_last > L:
                    raise ValueError(f"last_epoch={last_epoch} exceeds history length {L}")

                start = max(0, effective_last - n_epochs)
                end   = effective_last

                residual_fold_means.append(np.mean(res_score[start:end]))
                residual_fold_stds.append(np.mean(res_std[start:end]))

                current_fold_successful = True

            except Exception as e:
                print(f"    Error loading Residual: {e}")
        else:
            print(f"    Warning: Missing residual file: {res_file.name}")

        # --- Mask History (Full Mask) ---
        mask_file = fold_path / MASK_HISTORY_FILE
        if current_fold_successful and mask_file.exists():
            try:
                mask_score = extract_dice_array(mask_file, MEAN_SCORE_KEY)
                mask_std   = extract_dice_array(mask_file, STD_SCORE_KEY)

                L = len(mask_score)
                effective_last = L if last_epoch is None else last_epoch
                if effective_last > L:
                    raise ValueError(f"last_epoch={last_epoch} exceeds history length {L}")

                start = max(0, effective_last - n_epochs)
                end   = effective_last

                mask_fold_means.append(np.mean(mask_score[start:end]))
                mask_fold_stds.append(np.mean(mask_std[start:end]))

                processed_folds += 1

            except Exception as e:
                print(f"    Error loading Mask: {e}. Skipping fold.")
                # Remove the data added from the residual loading step
                residual_fold_means.pop()
                residual_fold_stds.pop()

        elif current_fold_successful:
            print(f"    Warning: Missing mask file: {mask_file.name}. Skipping fold.")
            # Remove the data added from the residual loading step
            residual_fold_means.pop()
            residual_fold_stds.pop()

    # ---------------------------------------------------------------------
    # If a single fold was picked, return its results directly
    # ---------------------------------------------------------------------
    if fold_index is not None:
        if processed_folds == 0:
            print("\nNo valid data in requested fold.")
            return None
        
        print("\n=== SINGLE-FOLD RESULT ===\n")
        print(f"Fold: {fold_paths[0].name}")
        print(f"Epochs averaged: {start}–{end}")

        print(f"Residual mean: {residual_fold_means[0]:.4f}")
        print(f"Residual std:  {residual_fold_stds[0]:.4f}")
        print(f"Mask mean:     {mask_fold_means[0]:.4f}")
        print(f"Mask std:      {mask_fold_stds[0]:.4f}")
        
        # Return the lists containing the single fold's data
        return residual_fold_means, residual_fold_stds, mask_fold_means, mask_fold_stds

    # --- Multi-fold summary ---
    if processed_folds == 0:
        print("\nAnalysis failed: zero folds processed.")
        return None

    final_res_mean  = np.mean(residual_fold_means)
    final_res_std   = np.std(residual_fold_means)
    final_mask_mean = np.mean(mask_fold_means)
    final_mask_std  = np.std(mask_fold_means)

    results_data = [
        ["Residual Mask (T=3)", f"{final_res_mean:.4f}", f"{final_res_std:.4f}", processed_folds],
        ["Full Mask (T=3)",     f"{final_mask_mean:.4f}", f"{final_mask_std:.4f}", processed_folds],
    ]

    print("\n" + "="*50)
    print(f"| FINAL K-FOLD CROSS-VALIDATION RESULTS ({processed_folds} FOLDS) |")
    print("="*50 + "\n")
    print(f"Scores averaged over epochs {start} to {end}.")

    try:
        # Assuming tabulate is imported in the final execution environment
        # Note: Added safety check for 'tabulate'
        from tabulate import tabulate
        print(tabulate(
            results_data,
            headers=["Channel", "Mean Dice (μ)", "Std Dev (σ)", "Folds Used"],
            tablefmt="fancy_grid"
        ))
    except (NameError, ImportError):
         print("Warning: 'tabulate' not imported. Printing raw data arrays for t-test usage.")
    
    # *** CRITICAL CORRECTION: Added the required return for the t-test data ***
    return residual_fold_means, residual_fold_stds, mask_fold_means, mask_fold_stds


# --- STATISTICAL ANALYSIS ---

def corrected_paired_ttest_nadeau(
    model_a_scores: List[float],
    model_b_scores: List[float],
    N_TOTAL: int,  # Total number of unique data samples (e.g., 66 eyes)
    K: int,        # Number of folds (e.g., 5)
    test_set_ratio: float = 0.2
) -> Tuple[float, float, float]:
    """
    Performs a corrected paired t-test (Nadeau & Bengio, 2003).
    
    Returns:
        (t_statistic, p_value, corrected_std)
    """
    if len(model_a_scores) != K or len(model_a_scores) != len(model_b_scores):
        raise ValueError(f"Score lists must contain {K} elements (one per fold).")
    
    # 1. Calculate the difference vector (d)
    d = np.array(model_a_scores) - np.array(model_b_scores)
    
    # 2. Calculate mean difference (d_bar)
    d_bar = np.mean(d)
    
    # 3. Calculate sample variance of the difference (sigma_sq)
    sigma_sq = np.var(d, ddof=1) 
    
    # 4. Calculate the correction factor: accounts for dependent test sets
    N_train_over_N_test = (1 - test_set_ratio) / test_set_ratio
    correction_factor = (1.0 / K) + (1.0 / (N_train_over_N_test * K))
    
    # 5. Calculate the corrected variance and standard deviation
    sigma_sq_corr = correction_factor * sigma_sq
    sigma_corr = np.sqrt(sigma_sq_corr)
    
    # 6. Calculate the corrected t-statistic
    t_stat = d_bar / sigma_corr
    
    # Degrees of freedom is K - 1
    df = K - 1
    
    # 7. Calculate the two-sided p-value
    p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df))
    
    return t_stat, p_value, sigma_corr

# ---------------------------------------------------------------------

def paired_ttest_student(
    model_a_scores: List[float],
    model_b_scores: List[float],
    K: int # Number of folds (sample size for difference vector)
) -> Tuple[float, float]:
    """
    Performs a standard paired Student's t-test on K-fold CV results.
    
    Returns:
        (t_statistic, p_value)
    """
    if len(model_a_scores) != K or len(model_a_scores) != len(model_b_scores):
        raise ValueError(f"Score lists must contain {K} elements (one per fold).")
    
    # 1. Calculate the difference vector (d)
    d = np.array(model_a_scores) - np.array(model_b_scores)
    
    # 2. Calculate mean difference (d_bar)
    d_bar = np.mean(d)
    
    # 3. Calculate the standard error of the mean difference (s_d / sqrt(K))
    # s_d (standard deviation of the difference) uses ddof=1 for sample SD
    std_err_diff = np.std(d, ddof=1) / np.sqrt(K)
    
    # 4. Calculate the t-statistic
    t_stat = d_bar / std_err_diff
    
    # Degrees of freedom is K - 1
    df = K - 1
    
    # 5. Calculate the two-sided p-value
    p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df))
    
    return t_stat, p_value

# ---------------------------------------------------------------------

def compare_models_nadeau_bengio(
    model_a_dir: str,
    model_b_dir: str,
    # Constants must be passed or defined globally where this is executed.
    RESIDUAL_HISTORY_FILE: str, MASK_HISTORY_FILE: str,
    MEAN_SCORE_KEY: str, STD_SCORE_KEY: str,
    T_CRITICAL_INDEX: int,
    N_TOTAL_SAMPLES: int = 66, 
    K_FOLDS: int = 5,
    n_epochs: int = 10,
    last_epoch: int = 60
) -> None:
    """
    Loads fold-wise scores for two models and performs the corrected paired t-test.
    
    NOTE: Assumes analyze_kfold_results and corrected_paired_ttest_nadeau are defined/accessible.
    """
    
    # --- 1. Load Scores for Model A ---
    model_a_name = Path(model_a_dir).name
    print(f"Loading scores for Model A: {model_a_name}...")
    
    # analyze_kfold_results must be the one defined in the previous response, 
    # which returns the raw residual_fold_means as the first item.
    results_a = analyze_kfold_results(
        model_a_dir,
        RESIDUAL_HISTORY_FILE, MASK_HISTORY_FILE,
        MEAN_SCORE_KEY, STD_SCORE_KEY,
        n_epochs=n_epochs,
        t_index=T_CRITICAL_INDEX,
        last_epoch=last_epoch
    )

    if results_a is None:
        print("Error: Could not load valid data for Model A.")
        return

    # Extract the raw Growth Mask (Residual) fold scores
    a_growth_scores = results_a[0]
    K_used_A = len(a_growth_scores)

    # --- 2. Load Scores for Model B ---
    model_b_name = Path(model_b_dir).name
    print(f"\nLoading scores for Model B: {model_b_name}...")
    results_b = analyze_kfold_results(
        model_b_dir,
        RESIDUAL_HISTORY_FILE, MASK_HISTORY_FILE,
        MEAN_SCORE_KEY, STD_SCORE_KEY,
        n_epochs=n_epochs,
        t_index=T_CRITICAL_INDEX,
        last_epoch=last_epoch
    )

    if results_b is None:
        print("Error: Could not load valid data for Model B.")
        return

    b_growth_scores = results_b[0]
    K_used_B = len(b_growth_scores)
    
    if K_used_A != K_used_B:
         print(f"\nError: Mismatched number of successful folds between models (A={K_used_A}, B={K_used_B}). Cannot run paired test.")
         return
         
    K_used = K_used_A
    
    # --- 3. Perform T-Test ---
    t_stat, p_value, sigma_corr = corrected_paired_ttest_nadeau(
        a_growth_scores, b_growth_scores,
        N_TOTAL=N_TOTAL_SAMPLES,
        K=K_FOLDS,
        test_set_ratio=(1.0 / K_FOLDS)
    )

    # --- 4. Report Results ---
    mean_diff = np.mean(a_growth_scores) - np.mean(b_growth_scores)
    
    print("\n" + "="*60)
    print("✨ CORRECTED PAIRED T-TEST (NADEAU & BENGIO) RESULTS ✨")
    print("="*60)
    print(f"Comparison: {model_a_name} vs. {model_b_name}") 
    print(f"Metric: Growth Mask DICE (Avg. last {n_epochs} epochs)")
    print("-" * 60)
    print(f"Folds used (K): {K_used} / {K_FOLDS}")
    print(f"Total Samples (N): {N_TOTAL_SAMPLES}")
    print("-" * 60)
    print(f"Mean Difference (A - B): {mean_diff:.4f}")
    print(f"Corrected t-statistic: {t_stat:.4f}")
    print(f"Degrees of Freedom (K-1): {K_used - 1}")
    print(f"Corrected p-value: {p_value:.6f}")
    print("-" * 60)
    
    if p_value < 0.05:
        print("Conclusion: The difference is STATISTICALLY SIGNIFICANT (p < 0.05).")
        if mean_diff > 0:
            print(f"Model A ({model_a_name}) significantly outperforms Model B.")
        else:
            print(f"Model B ({model_b_name}) significantly outperforms Model A.")
    else:
        print("Conclusion: The difference is NOT STATISTICALLY SIGNIFICANT (p > 0.05).")
        print("The performance difference could be due to random chance/experimental variation.")
        
        
# ---------------------------------------------------------------------------------------

# --- MAIN COMPARISON FUNCTION (Uses the Student's t-test) ---

def compare_models(
    model_a_dir: str,
    model_b_dir: str,
    # Constants must be passed or defined globally where this is executed.
    RESIDUAL_HISTORY_FILE: str, MASK_HISTORY_FILE: str,
    MEAN_SCORE_KEY: str, STD_SCORE_KEY: str,
    T_CRITICAL_INDEX: int,
    N_TOTAL_SAMPLES: int = 66, 
    K_FOLDS: int = 5,
    n_epochs: int = 10,
    last_epoch: int = 60
) -> None:
    """
    Loads fold-wise scores for two models and performs the standard paired Student's t-test.
    """
    
    # --- 1. Load Scores for Model A ---
    model_a_name = Path(model_a_dir).name
    print(f"Loading scores for Model A: {model_a_name}...")
    
    # analyze_kfold_results must be defined and return the raw fold score lists.
    results_a = analyze_kfold_results(
        model_a_dir,
        RESIDUAL_HISTORY_FILE, MASK_HISTORY_FILE,
        MEAN_SCORE_KEY, STD_SCORE_KEY,
        n_epochs=n_epochs,
        t_index=T_CRITICAL_INDEX,
        last_epoch=last_epoch
    )

    if results_a is None:
        print("Error: Could not load valid data for Model A.")
        return

    # Extract the raw Growth Mask (Residual) fold scores
    a_growth_scores = results_a[0]
    K_used_A = len(a_growth_scores)

    # --- 2. Load Scores for Model B ---
    model_b_name = Path(model_b_dir).name
    print(f"\nLoading scores for Model B: {model_b_name}...")
    results_b = analyze_kfold_results(
        model_b_dir,
        RESIDUAL_HISTORY_FILE, MASK_HISTORY_FILE,
        MEAN_SCORE_KEY, STD_SCORE_KEY,
        n_epochs=n_epochs,
        t_index=T_CRITICAL_INDEX,
        last_epoch=last_epoch
    )

    if results_b is None:
        print("Error: Could not load valid data for Model B.")
        return

    b_growth_scores = results_b[0]
    K_used_B = len(b_growth_scores)
    
    if K_used_A != K_used_B:
         print(f"\nError: Mismatched number of successful folds between models (A={K_used_A}, B={K_used_B}). Cannot run paired test.")
         return
         
    K_used = K_used_A
    
    # --- 3. Perform Standard Paired T-Test ---
    t_stat, p_value = paired_ttest_student(
        a_growth_scores, b_growth_scores,
        K=K_used
    )

    # --- 4. Report Results ---
    mean_diff = np.mean(a_growth_scores) - np.mean(b_growth_scores)
    
    print("\n" + "="*60)
    print("✨ STANDARD PAIRED STUDENT'S T-TEST RESULTS (UNCORRECTED) ✨")
    print("="*60)
    print(f"Comparison: {model_a_name} vs. {model_b_name}") 
    print(f"Metric: Growth Mask DICE (Avg. last {n_epochs} epochs)")
    print("-" * 60)
    print(f"Folds used (K): {K_used} / {K_FOLDS}")
    print(f"Total Samples (N): {N_TOTAL_SAMPLES}")
    print("-" * 60)
    print(f"Mean Difference (A - B): {mean_diff:.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"Degrees of Freedom (K-1): {K_used - 1}")
    print(f"p-value: {p_value:.6f}")
    print("-" * 60)
    
    if p_value < 0.05:
        print("Conclusion: The difference is STATISTICALLY SIGNIFICANT (p < 0.05).")
        if mean_diff > 0:
            print(f"Model A ({model_a_name}) significantly outperforms Model B.")
        else:
            print(f"Model B ({model_b_name}) significantly outperforms Model A.")
    else:
        print("Conclusion: The difference is NOT STATISTICALLY SIGNIFICANT (p > 0.05).")
        print("The performance difference could be due to random chance/experimental variation.")