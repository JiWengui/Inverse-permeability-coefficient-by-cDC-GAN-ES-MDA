import numpy as np
import Mod_my
import os
import plotting

w_all = [0.5]
rr_all = [0.6]
alpha_geo_all = [1.05]

for w_num in w_all:
    for rr_num in rr_all:
        for alpha_geo_num in alpha_geo_all:
            print('w:{}; rr:{}; alpha_geo:{}'.format(w_num,rr_num,alpha_geo_num))
            N_iter = 20 # Number of iterations
            alpha_geo = alpha_geo_num
            w=w_num
            rr=rr_num
            inflation = 'y'
            from InputSettings import (ref_solution, last_iter_pred)
            space_transform = 'y'
            localize = 'n'
            iter_loc='y'
            loc_time='n'
            loc_space = 'y'
            Obs_file = np.loadtxt('./obs_ions.txt', dtype=float)    # Load observation data # normalization
            x_obs = Obs_file[:, 0]
            y_obs = Obs_file[:, 1]
            z_obs = Obs_file[:, 2]
            time_obs = Obs_file[:, 3]
            Obs = np.atleast_2d(Obs_file[:, 4]).T[:1570 - 314,:]
            N_obs = Obs.shape[0]
            u_weight = []
            u_fuyuan = []
            for i in range(314):
                if i % 2 == 0:
                    u_weight.append(0.5)#2
                    u_fuyuan.append(2.)#0.5
                else:
                    u_weight.append(1)#4
                    u_fuyuan.append(1)#0.25
            u_weight = np.array(u_weight).reshape(314,1)
            u_fuyuan = np.array(u_fuyuan)
            fu_50 = np.ones((314, 50))
            for j in range(50):
                fu_50[:, j] = u_fuyuan
            weight = np.concatenate((u_weight,np.ones((1570-314, 1))*0.5), axis=0)
            fuyuan_weight = np.concatenate((fu_50, np.ones((1570 - 314, 50)) * 2), axis=0)
            Obs = Obs * weight[:1570 - 314,:]
            True_par_file = np.loadtxt('./data/par.txt', dtype=float)  # Real parameters # "NaN"
            x_par = True_par_file[:, 0]
            y_par = True_par_file[:, 1]
            z_par = True_par_file[:, 2]
            time_par = True_par_file[:, 3]
            True_par = True_par_file[:, 4]
            X = np.loadtxt('./data/ens_no_normal.txt')/20
            ens = X.shape[1]
            N_par = X.shape[0]

            new_err = 'y'   # Whether to generate new random errors
            if new_err == 'n':
                R = np.loadtxt('R.txt')
                eps = np.loadtxt('Errors.txt')
            elif new_err == 'y':
                from InputSettings import Func_err
                eps, R = Func_err(Obs, ens)
            else:
                print('Error, invalid input')

            al_i = np.ones((N_iter, 1), float)
            for i in range(1, N_iter):
                al_i[i] = al_i[i - 1] / alpha_geo
            sum_al_i = sum(1. / al_i)
            alpha = al_i * sum_al_i
            sum_alpha = sum(1. / alpha)

            if localize == 'y':
                from my_setting import localization
            if localize == 'y' and iter_loc == 'n':
                (rho_yy, rho_xy, rho_xx) = localization(X, True_par_file[:, 0:4], Obs_file[:, 0:4], loc_space, loc_time, iter_loc)

            r = []
            pred = np.zeros((N_obs, ens))
            Xprev = np.copy(X)
            for i in range(0, N_iter):
                print('Iteration ' + str(i + 1))
                R_corr = alpha[i] * R
                r.append(R_corr[i, i])
                if localize == 'y' and iter_loc == 'y':
                    (rho_yy, rho_xy, rho_xx) = localization(Xprev, True_par_file[:, 0:4], Obs_file[:, 0:4], loc_space, loc_time,
                                                            iter_loc)

                for j in range(0, ens):
                    pred[:, j] = Mod_my.forword_model(Xprev[:, j], )    # call forward model
                a = 0
                pred = pred * weight[:1570 - 314, :]
                Xprev = np.where(Xprev < 0.01, 0.01, Xprev)  # lower bounds

                if space_transform == 'y':
                    from InputSettings import forward_transf
                    Xprev = forward_transf(Xprev)

                xm = np.atleast_2d(Xprev.mean(1)).T
                ym = np.atleast_2d(pred.mean(1)).T
                Qx = Xprev - xm * np.ones((1, ens))
                Qy = pred - ym * np.ones((1, ens))
                Qxy = Qx @ Qy.T / (ens - 1)
                Qyy = Qy @ Qy.T / (ens - 1)
                Qxx = Qx @ Qx.T / (ens - 1)
                if localize == 'y':
                    Qxy = rho_xy * Qxy
                    Qyy = rho_yy * Qyy
                    Qxx = rho_xx * Qxx

                Gain = Qxy @ np.linalg.inv(Qyy + R_corr)

                Xnew = Xprev + Gain @ (Obs @ np.ones((1, ens)) + (alpha[i]) ** (1 / 2) * eps - pred)    # update
                Xnew = (1 - w) * Xnew + w * Xprev

                if space_transform == 'y':
                    from InputSettings import backward_transf
                    Xnew = backward_transf(Xnew)
                    Xprev = backward_transf(Xprev)

                if inflation == 'y':
                    Xnew_mean = np.atleast_2d(np.mean(Xnew, axis=1)).T
                    Xnew = Xnew_mean * np.ones((1, ens)) + rr * (Xnew - Xnew_mean * np.ones((1, ens)));

                """Apply bounds constraints to the adjusted parameters."""
                m_bounds = [0.01, 100]
                Xnew = np.where(Xnew < m_bounds[0], m_bounds[0], Xnew)  # lower bounds
                Xnew = np.where(Xnew > m_bounds[1], m_bounds[1], Xnew)  # upper bounds

                Xprev = np.copy(Xnew)
                Xp = np.mean(Xprev, axis=1)
                pred_mean = np.mean(pred, axis=1)

                if ref_solution == 'y':
                    from InputSettings import Metrics_obs_par
                    metrics_iter = Metrics_obs_par(Xprev, pred, True_par, Obs)
                else:
                    from InputSettings import Metrics_obs
                    metrics_iter = Metrics_obs(Xprev, pred, Obs)

                if i == 0:
                    metrics_name = list(metrics_iter.keys())
                    metrics_dict = metrics_iter.copy()
                else:
                    for m in metrics_name:
                        metrics_dict[m] = metrics_dict[m] + metrics_iter[m]

                if i == 0:
                    with open('./output/_Xprev_iter0.txt', 'wb') as f:   # Write initial parameter
                        np.savetxt(f, X, fmt='%.9f')

                if i % 1 == 0:
                    with open('./output/_Xprev_iter' + str(i + 1) + '.txt', 'wb') as f:  # Write parameter
                        np.savetxt(f, Xprev, fmt='%.9f', newline='\r\n')
                    with open('./output/_pred_iter' + str(i + 1) + '.txt', 'wb') as f:   # Write the predicted values in the iteration
                        np.savetxt(f, pred*fuyuan_weight[:1570 - 314, :], fmt='%.9f', newline='\n')
            plotting.main(N_iter=N_iter, w_num=w_num, rr_num=rr_num, alpha_geo_num=alpha_geo_num,)