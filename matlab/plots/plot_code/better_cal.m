% 元学习优于线性阻力模型
liner_vx_rmse = 1.546;
meta_vx_rmse = 0.805;
pure_vx_rmse = 2.072;

meta_liner = (liner_vx_rmse - meta_vx_rmse)/liner_vx_rmse * 100
meta_pure = (pure_vx_rmse - meta_vx_rmse)/pure_vx_rmse * 100

% 8.5 + sint
meta_vx_rmse = 0.603;
pure_vx_rmse = 1.470;
meta_pure = (pure_vx_rmse - meta_vx_rmse)/pure_vx_rmse * 100
