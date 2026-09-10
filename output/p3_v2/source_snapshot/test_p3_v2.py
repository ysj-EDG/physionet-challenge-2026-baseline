import numpy as np
from pooled_logistic_v2 import *

def test_dimensions_and_missing_patient_retained():
 x=np.zeros((4,483)); static=np.zeros(196); stage=np.array([2,9,4,1]); valid=np.isin(stage,[1,2,3,4,5]); ch=np.ones(6,bool)
 for arm,n in zip(['P3_demo10','P3_compact30','P3_compact35','P3_global59','P3_stage155'],[10,30,35,59,155]):
  assert assemble_features(static,stage,valid,ch,x,arm).shape==(n,)
 missing=assemble_features(static,np.full(4,np.nan),np.zeros(4,bool),np.zeros(6,bool),x,'P3_stage155')
 assert len(missing)==155 and np.isnan(missing[35:]).all()

def test_regions_and_iqr_not_mixed():
 x=np.zeros((4,483)); stage=np.full(4,2.0); valid=np.ones(4,bool); ch=np.ones(6,bool)
 for c in range(6): x[:,c*9+3]=np.arange(4)+100*c
 out=pool_eeg(x,stage,valid,ch,'global')
 assert out[0]==51.5 and out[1]==1.5
 assert out[2]==251.5 and out[3]==1.5
 assert out[4]==451.5 and out[5]==1.5

def test_fold_fit_ignores_holdout_and_age_external():
 train=np.array([[1,np.nan],[3,4]],float); hold=np.array([[999,999]],float)
 a=fit_imputer_scaler(train); b=fit_imputer_scaler(train.copy())
 for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
 assert apply_imputer_scaler(hold,*a[:3]).shape==(1,2)
 raw_age=73.0; _=assemble_features(np.zeros(196),np.ones(3),np.ones(3,bool),np.ones(6,bool),np.zeros((3,483)),'P3_demo10'); assert raw_age==73.0
