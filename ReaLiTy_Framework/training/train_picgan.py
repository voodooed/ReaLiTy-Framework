import os
import sys
import importlib

class PICGANTrainer:
    def __init__(self, reality_config):
        """
        reality_config: dict containing:
            - mode: "sensor" or "weather"
            - picgan_root: path to PICGAN_1.0 directory
            - experiment_name: string (e.g., T1)
            - num_epochs: optional override
        """
        self.mode = reality_config["mode"]
        self.picgan_root = reality_config["picgan_root"]
        self.experiment_name = reality_config["experiment_name"]

        # 1. Ensure PICGAN root is in path
        if self.picgan_root not in sys.path:
            sys.path.insert(0, self.picgan_root)

        # 2. Force load/reload the config module
        if "config" in sys.modules:
            importlib.reload(sys.modules["config"])
        self.picgan_config = importlib.import_module("config")

        self._prepare_weight_dirs()
        
        # 3. Apply overrides from reality_config if they exist
        if "num_epochs" in reality_config:
            self.picgan_config.NUM_EPOCHS = reality_config["num_epochs"]

    def _prepare_weight_dirs(self):
        # Follow your "Important Structural Improvement" rule: 
        # Source -> Target naming for weight files
        weight_root = os.path.join("weights", self.mode)
        os.makedirs(weight_root, exist_ok=True)
        self.weight_root = weight_root

        # Redirecting all checkpoint paths in the PICGAN config object
        self.picgan_config.CHECKPOINT_GEN_S = os.path.join(weight_root, f"{self.experiment_name}_gen_S.pth.tar")
        self.picgan_config.CHECKPOINT_GEN_R = os.path.join(weight_root, f"{self.experiment_name}_gen_R.pth.tar")
        self.picgan_config.CHECKPOINT_DISC_S = os.path.join(weight_root, f"{self.experiment_name}_disc_S.pth.tar")
        self.picgan_config.CHECKPOINT_DISC_R = os.path.join(weight_root, f"{self.experiment_name}_disc_R.pth.tar")

        self.picgan_config.OUTPUT_FOLDER = os.path.join(weight_root, "validation_samples")
        os.makedirs(self.picgan_config.OUTPUT_FOLDER, exist_ok=True)

    def run(self):
        print(f"\n ReaLiTy PICGAN Wrapper: Training Start")
        print(f"Target Mode: {self.mode} | Weight Dir: {self.weight_root}")

        # Import the main script and execute
        # We use importlib here so it picks up our modified 'config' from sys.modules
        main_module = importlib.import_module("main")
        
        try:
            main_module.main()
            print("\nPICGAN Training Cycle Finished.")
        except Exception as e:
            print(f"\n Training crashed: {str(e)}")
            raise e
