print(df['phi0'][:])
i = 1

Cx = df['xc'].iloc[i]
Cy = df['yc'].iloc[i]
r = df['R'].iloc[i]

angle = df['phi0'].iloc[i]
delta_angle = df['dphi'].iloc[i]

X = Cx + (r * np.cos(angle))
Y = Cy + (r * np.sin(angle))

X_final = Cx + (r * np.cos(angle + delta_angle))
Y_final = Cy + (r * np.sin(angle + delta_angle))

print(X, Y)
print(df['x0_true'].iloc[i], df['y0_true'].iloc[i])

print(X_final, Y_final)
print(df['xf_true'].iloc[i], df['yf_true'].iloc[i])