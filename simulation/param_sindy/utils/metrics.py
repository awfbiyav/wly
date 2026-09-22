import numpy as np

def compute_metrics(result):
    model=result.get('model') or {}
    return {
        'success':int(bool(result.get('success',False))),
        'terminal_error':float(result.get('terminal_error',np.nan)),
        'min_dist':float(result.get('min_obstacle_distance',np.nan)),
        'arrival_time_success':float(result.get('arrival_time_success',np.nan)),
        'control_energy':float(result.get('control_energy',np.nan)),
        'update_count':int(result.get('update_count',0)),
        'accepted_update_count':int(result.get('accepted_update_count',0)),
        'confidence':float(model.get('confidence',np.nan)),
        'alpha':float(model.get('alpha',np.nan)),
        'pred_gain':float(model.get('pred_gain',np.nan)),
        'recent_gain':float(model.get('recent_gain',np.nan)),
        'norm_id_error':float(model.get('norm_id_error',np.nan)),
        'nnz_model':int(model.get('n_nonzero',0)),
        'alpha_mean':float(np.nanmean(result.get('alpha_history',[]))) if result.get('alpha_history') else np.nan,
        'alpha_std':float(np.nanstd(result.get('alpha_history',[]))) if result.get('alpha_history') else np.nan,
        'regime_score_final': float(result.get('regime_score_history',[])[-1]) if result.get('regime_score_history') else np.nan,
        'regime_alarm_rate': float(np.nanmean(result.get('regime_alarm_history',[]))) if result.get('regime_alarm_history') else np.nan,
    }

def percentile_no_nan(x,p):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    return float(np.percentile(x,p)) if len(x) else np.nan
