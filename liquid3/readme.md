# 项目理论基础

## 1.方程

$$
\rho_i\frac{\mathrm{D}v_i}{\mathrm{D}t}=-\nabla p+\eta\nabla^2 v_i+f_{ext}
$$

第一步，忽略压强

$$
\rho_i=\sum_j m_j W_{ij}(r)\\
v_i^*=v_i^n+(\frac{f_{ext}}{\rho_i}+\frac{2\eta}{\rho_i}\sum_j\frac{m_j}{\rho_j}\frac{W'(r)}{r}(v_i-v_j))\Delta t
$$

第二步，修正压强

$$
\sum_j \frac{m_j}{\rho_j}(\frac{1}{\rho_i}+\frac{1}{\rho_j})(p_i-p_j)\frac{W'(r)}{r}=-\frac{1}{\Delta t}\sum_j \frac{m_j}{\rho_j}(v_j^*\cdot \frac{r_i-r_j}{r})W'
$$

第三步，修正速度

$$
v^{n+1}_i=v_i^*-\Delta t\sum_j m_j(\frac{p_i}{\rho_i^2}+\frac{p_j}{\rho_j^2})\frac{r_i-r_j}{r}W'_{ij}(r)
$$