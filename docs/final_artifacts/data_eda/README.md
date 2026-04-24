# Data EDA

This summary reflects the final tested AutoNav dataset snapshot used for repository validation.

## Dataset scope

- Run folders discovered: 17
- Total raw rows across archived dataset CSV files: 9,536
- Non-empty runs: 16
- Empty runs: 1

## Sensor coverage highlights

- CAM0/RGB coverage is effectively complete in the archived set.
- CAM1, IR, and depth are present in a subset of rows.
- IMU columns are absent in this archived snapshot.
- 
<img width="1335" height="735" alt="sensor_coverage" src="https://github.com/user-attachments/assets/02167077-4b3f-4e75-8509-9091843b94b7" />


## Distribution plots

<img width="1185" height="735" alt="steering_distribution" src="https://github.com/user-attachments/assets/c393b628-b04c-4d6f-881a-c8cac0086cac" />

<img width="1185" height="735" alt="throttle_distribution" src="https://github.com/user-attachments/assets/3a95d2cd-91b2-43fd-8932-00b3a3dece7b" />


## Representative sample and preprocessing

<img width="1784" height="1036" alt="sample_sensor_grid" src="https://github.com/user-attachments/assets/99f989fb-b658-4cfb-8fa9-20a629955bbf" />

<img width="1485" height="646" alt="preprocessing_example" src="https://github.com/user-attachments/assets/8de97f3e-9117-4f05-a144-ca55bd29d3e3" />


## Raw table artifact

The detailed per-column coverage table is preserved in [data_summary.csv](data_summary.csv).
