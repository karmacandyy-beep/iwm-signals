    # --- Signal thresholds ---
    opening_range_minutes: int = 15
    entry_window_start: dtime = dtime(9, 45)
    entry_window_end: dtime = dtime(15, 30)
    hard_time_stop: dtime = dtime(15, 45)
    rvol_threshold: float = 1.5
    vwap_confirmation: bool = True