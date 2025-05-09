import numpy as np

class STAR(object):
    def __init__(self,
                 num_antennas,
                 num_star_elements,
                 num_users,
                 num_d2d_pairs,
                 channel_est_error = False,
                 awgn_var = 1e-2,
                 channel_noise_var = 1e-2):

        self.M = num_antennas
        self.N = num_star_elements
        self.K = num_users
        self.D = num_d2d_pairs

        self.channel_est_error = channel_est_error
        self.awgn_var = awgn_var
        self.channel_noise_var = channel_noise_var

        self.action_dim = 2 * self.M * self.K + 2 * self.N ** 2
        self.state_dim = self.action_dim + 2 * (self.M * self.K + self.M * self.N + self.N * self.K + 2 * self.M * self.D + self.D ** 2 + 2 * self.N * self.D + 2 * self.D * self.K)

        self.bs_users = None
        self.bs_star = None
        self.star_users = None

        self.bs_d2d = None
        self.d2d_d2d = None
        self.star_d2d = None
        self.d2d_users = None

        self.G = np.random.randn(self.M, self.K) + np.random.randn(self.M, self.K) *1j
        trace_GGH = np.trace(self.G @ self.G.conj().T)
        scaling_factor = np.sqrt(self.K/trace_GGH)
        self.G = scaling_factor * self.G

        self.Phi = np.eye(self.N, dtype=complex)

        self.state = None
        self.done = False

        self.episode_t = None

    def _compute_tile(self, matrix):
        return matrix.T @ self.Phi @ self.bs_star @ self.G

    def compute_energy(self, matrix):
        return np.sum(np.abs(matrix)**2)

    def stack_matrix(self, matrix):
        return np.hstack((np.real(matrix).ravel(),np.imag(matrix).ravel()))

    def reset(self):
        self.episode_t = 0

        #Intial users matrix
        self.bs_users = np.random.normal(0,np.sqrt(0.5), (self.M, self.K)) + 1j * np.random.normal(0,np.sqrt(0.5), (self.M, self.K))
        self.bs_star = np.random.normal(0,np.sqrt(0.5), (self.M, self.N)) + 1j * np.random.normal(0,np.sqrt(0.5), (self.M, self.N))
        self.star_users = np.random.normal(0,np.sqrt(0.5), (self.N, self.K)) + 1j * np.random.normal(0,np.sqrt(0.5), (self.N, self.K))

        # print(self.bs_users)
        # print(self.bs_star)
        # print(self.star_users)

        init_action_G = np.hstack((np.real(self.G.reshape(-1)),np.imag(self.G.reshape(-1))))
        init_action_Phi = np.hstack((np.real(self.Phi.reshape(-1)),np.imag(self.Phi.reshape(-1))))
        init_action = np.hstack((init_action_G,init_action_Phi))

        #Initial d2d matrix
        self.bs_d2d = np.random.normal(0,np.sqrt(0.5), (self.M, 2 * self.D )) + 1j * np.random.normal(0,np.sqrt(0.5), (self.M, 2 * self.D))
        self.d2d_d2d = np.random.normal(0,np.sqrt(0.5), (self.D, self.D)) + 1j * np.random.normal(0,np.sqrt(0.5), (self.D, self.D))
        # First d row is d2d_star the others is star_d2d
        self.star_d2d = np.random.normal(0,np.sqrt(0.5), (self.N, 2 * self.D)) + 1j * np.random.normal(0,np.sqrt(0.5), (self.N, 2* self.D))
        self.d2d_users = np.random.normal(0,np.sqrt(0.5), (2 * self.D, self.K)) + 1j * np.random.normal(0,np.sqrt(0.5), (2 * self.D, self.K))

        # print(self.bs_d2d)
        # print(self.d2d_d2d)
        # print(self.star_d2d)
        # print(self.d2d_users)
        self.state = np.hstack((init_action, self.stack_matrix(self.bs_users), self.stack_matrix(self.bs_star),self.stack_matrix(self.star_users),
                                self.stack_matrix(self.bs_d2d), self.stack_matrix(self.d2d_d2d), self.stack_matrix(self.star_d2d), self.stack_matrix(self.d2d_users)))

        return self.state

    def compute_reward(self, Phi):
        diag_Phi = np.diag(Phi)
        diag_Phi1 = np.zeros((self.N,), dtype=complex)
        diag_Phi2 = np.zeros((self.N,), dtype=complex)
        diag_Phi1[:self.N//2] = diag_Phi[:self.N//2]
        diag_Phi2[self.N//2:] = diag_Phi[self.N//2:]

        Phi1 = np.diag(diag_Phi1)
        Phi2 = np.diag(diag_Phi2)

        reward = 0
        opt_reward = 0
        min_R_d2d = 100

        # d2d_users_t = self.d2d_users[:self.D,:]
        # d2d_star_t = self.star_d2d[:self.N,:]


        for k in range(self.K):
            bs_user_k = self.bs_users[:,k]
            star_user_k = self.star_users[:,k]
            G_remove = np.delete(self.G,k,1)
            d2d_user_k = self.d2d_users[:self.D,k]

            if k < self.K // 2:
                Phi_k = Phi1
            else:
                Phi_k = Phi2


            x = self.compute_energy(bs_user_k @ self.G[:,k]) + self.compute_energy(star_user_k.T @ Phi_k @ self.bs_star @ self.G[:,k])
            interferences_users = self.compute_energy(bs_user_k @ G_remove) + self.compute_energy(
                star_user_k.T @ Phi_k @ self.bs_star @ G_remove)
            interferences_d2d = self.compute_energy(d2d_user_k) + self.compute_energy(
                star_user_k.T @ Phi_k @ self.star_d2d[:, :self.D])
            x = x.item()
            y = interferences_users + interferences_d2d

            rho_k = x / y
            #print(np.log(1 + rho_k)/ np.log(2))

            reward += np.log(1 + rho_k)/ np.log(2)
            opt_reward += np.log(1 + self.K/2)/ np.log(2)

        for j in range(self.D):
            d2d_remove = np.delete(self.d2d_d2d,j,0)
            d2d_star_remove  = np.delete(self.star_d2d[:, : self.D],j,1)

            if j < self.D // 2:
                Phi_j = Phi1
            else:
                Phi_j = Phi2

            x = self.compute_energy(self.d2d_d2d[j,j]) + self.compute_energy(self.star_d2d[:,self.D + j] @ self.Phi @ self.star_d2d[:,j].T)
            x = x.item()

            interferences_users = self.compute_energy(self.bs_d2d[:,j] @ self.G) + self.compute_energy(
                self.star_d2d[:,self.D + j].T @ Phi_j @ self.bs_star @ self.G)
            interferences_d2d = self.compute_energy(d2d_remove[:,j]) + self.compute_energy(
                self.star_d2d[:,self.D + j].T @ Phi_j @ d2d_star_remove)

            rho_j = x / interferences_users + interferences_d2d
            achievable_rate = np.log(1 + rho_j)/ np.log(2)
            if min_R_d2d > achievable_rate: min_R_d2d = achievable_rate


        return reward, opt_reward, min_R_d2d

    def step(self, action):
        self.episode_t += 1

        action = action.reshape(self.action_dim)
        G_real = action[:self.M ** 2]
        G_imag = action[self.M ** 2:2 * self.M ** 2]

        Phi_real = action[-2 * self.N:-self.N]
        Phi_imag = action[-self.N:]

        self.G = G_real.reshape(self.M, self.K) + 1j * G_imag.reshape(self.M, self.K)
        trace_GGH = np.trace(self.G @ self.G.conj().T)
        self.G = self.G * np.sqrt(self.K / trace_GGH)

        self.Phi = Phi_real + 1j * Phi_imag
        for i in range(len(self.Phi)):
            self.Phi[i] = self.Phi[i] / np.abs(self.Phi[i])
        self.Phi = np.eye(self.N, dtype=complex) * (self.Phi)

        # h_t_tilde = self._compute_tilde(self.h_t)
        # h_r_tilde = self._compute_tilde(self.h_r)
        #
        # power_r = np.linalg.norm(h_t_tilde, axis=0).reshape(1, -1) ** 2 + np.linalg.norm(h_r_tilde, axis=0).reshape(1,
        #                                                                                                             -1) ** 2
        #
        # H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
        # h_t_real, h_t_img = np.real(self.h_t).reshape(1, -1), np.imag(self.h_t).reshape(1, -1)
        # h_r_real, h_r_img = np.real(self.h_r).reshape(1, -1), np.imag(self.h_r).reshape(1, -1)

        reward, opt_reward, min_R_d2d = self.compute_reward(self.Phi)

        self.state = np.hstack((action, self.stack_matrix(self.bs_users), self.stack_matrix(self.bs_star), self.stack_matrix(self.star_users),
                                self.stack_matrix(self.bs_d2d),self.stack_matrix(self.d2d_d2d),self.stack_matrix(self.star_d2d),self.stack_matrix(self.d2d_users)))



        done = opt_reward == reward

        return self.state, reward, done, None

    def close(self):
        pass

# if __name__ == '__main__':
#     object = STAR(4,4,4,4)
#     object.reset()
#     print(object.compute_reward(np.eye(4)))
