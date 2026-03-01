import pandas as pd
from src.monitoring.drift_detection import check_data_drift

ref = pd.DataFrame({"a": [1,2,3,4,5]})
cur = pd.DataFrame({"a": [10,11,12,13,14]})

print(check_data_drift(ref, cur))