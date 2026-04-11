import torch
import config
from tqdm import tqdm

# S - Simulated Domain
# R - Real Domain


def train_fn(
    disc_S, disc_R, gen_R, gen_S, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    R_reals = 0
    R_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (sim, real, phy) in enumerate(loop):
        sim = sim.to(config.DEVICE) # 3 channels - Depth, IA, Reflectance
        real = real.to(config.DEVICE) # 1 channel - Real intensity
        phy = phy.to(config.DEVICE) # 1 channel - Physics-based intensity

        # Train Discriminators R and S
        with torch.cuda.amp.autocast():  #For float16 training
            
            
            fake_real = gen_R(sim)
            D_R_real = disc_R(real)
            D_R_fake = disc_R(fake_real.detach())
            R_reals += D_R_real.mean().item()
            R_fakes += D_R_fake.mean().item()
            D_R_real_loss = mse(D_R_real, torch.ones_like(D_R_real))
            D_R_fake_loss = mse(D_R_fake, torch.zeros_like(D_R_fake))
            D_R_loss = D_R_real_loss + D_R_fake_loss
            
            fake_sim = gen_S(real)
            D_S_real = disc_S(sim)
            D_S_fake = disc_S(fake_sim.detach())
            D_S_real_loss = mse(D_S_real, torch.ones_like(D_S_real))
            D_S_fake_loss = mse(D_S_fake, torch.zeros_like(D_S_fake))
            D_S_loss = D_S_real_loss + D_S_fake_loss

            # put it together
            D_loss = (D_R_loss + D_S_loss) / 2
          

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators R and S
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_S_fake = disc_S(fake_sim)
            D_R_fake = disc_R(fake_real)
            loss_G_S = mse(D_S_fake, torch.ones_like(D_S_fake))
            loss_G_R = mse(D_R_fake, torch.ones_like(D_R_fake))

            # cycle loss
            cycle_real = gen_R(fake_sim)
            cycle_sim = gen_S(fake_real)
            cycle_real_loss = l1(real, cycle_real)
            cycle_sim_loss = l1(sim, cycle_sim)
            
            """
            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_real = gen_R(real)
            identity_sim = gen_S(sim)
            identity_real_loss = l1(real, identity_real)
            identity_sim_loss = l1(sim, identity_sim)
            """
            #Physics Loss
            phy_loss = l1(fake_real, phy) #Loss between generated intensity(real domain) with physics-based intensity

            # add all together
            G_loss = (
                loss_G_R
                + loss_G_S
                + cycle_real_loss * config.LAMBDA_CYCLE
                + cycle_sim_loss * config.LAMBDA_CYCLE
               #+ identity_real_loss * config.LAMBDA_IDENTITY
               #+ identity_sim_loss * config.LAMBDA_IDENTITY
                + phy_loss * config.LAMBDA_Physics
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()


        loop.set_postfix(R_real=R_reals / (idx + 1), R_fake=R_fakes / (idx + 1))