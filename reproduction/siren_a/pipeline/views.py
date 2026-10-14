"""Frozen 807-unique-feature to five-view mapping."""
import numpy as np


def get_views(values,masks,manifest):
    values=np.asarray(values,dtype=np.float64); masks=np.asarray(masks,dtype=bool)
    if values.ndim!=2 or values.shape!=masks.shape or values.shape[1]!=807: raise ValueError("unique matrix shape")
    by_id={uid:i for i,uid in enumerate(manifest.unique_ids)}; vv={}; vm={}
    for view,size in manifest.view_sizes.items():
        rows=sorted((r for r in manifest.rows if r["view_id"]==view),key=lambda r:int(r["view_position"]))
        positions=np.asarray([by_id[r["unique_feature_id"]] for r in rows],dtype=np.int64)
        if len(positions)!=size: raise ValueError("view size")
        vv[view]=values[:,positions].copy(); vm[view]=masks[:,positions].copy()
    return vv,vm
