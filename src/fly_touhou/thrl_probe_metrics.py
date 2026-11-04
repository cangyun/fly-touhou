"""Diagnostic scoring only: preserve actual endpoints, cap comparison scores."""


def endpoint_fields(native_frames,frame_cap):
    if (type(native_frames) is not int or native_frames<0
            or type(frame_cap) is not int or frame_cap<=0):
        raise ValueError('Nonnegative measured frames and positive declared cap required')
    return dict(diagnostic_native_frame_cap=frame_cap,capped_score_native_frames=min(native_frames,frame_cap),
        native_endpoint_overshoot_frames=max(0,native_frames-frame_cap))


def score_frames(row,frame_cap=None):
    declared=row.get('diagnostic_native_frame_cap')
    if frame_cap is None:frame_cap=declared
    if frame_cap is None:return row['native_frames']  # Unmodified historical rows.
    if declared is not None and declared!=frame_cap:raise ValueError('Diagnostic score cap differs from the declared native protocol')
    fields=endpoint_fields(row['native_frames'],frame_cap)
    if ('capped_score_native_frames' in row and row['capped_score_native_frames']!=fields['capped_score_native_frames']):
        raise ValueError('Reported capped score differs from the actual endpoint')
    return fields['capped_score_native_frames']
