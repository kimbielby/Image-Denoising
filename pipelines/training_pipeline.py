from utils import *
from imports import *
from preprocessing import *
from dataloaders import *
from models import *
from configs import Config
from typing import Optional

class TrainingPipeline:
    """
    Complete training pipeline from preprocessing to model training.
    """
    def __init__(self, config: Config) -> None:
        """
        Initialise training pipeline.

        Args:
            config: Configuration object
        """
        # Load config
        self.config = config

        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Data attributes
        self.gt_train_list = None
        self.noisy_train_list = None
        self.gt_val_list = None
        self.noisy_val_list = None
        self.gt_test_list = None
        self.noisy_test_list = None

        # Loader attributes
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Model attributes
        self.model = None
        self.psnr_metric = None
        self.ssim_metric = None
        self.val_function = None

        # Results attributes
        self.history = None
        self.checkpoint_dir = None

    def run_preprocessing(self) -> None:
        """
        Run complete preprocessing pipeline.

        Handles image cropping, dataset splitting and augmentation. Updates
        internal state with preprocessed data splits.

        Steps:
        1. Get filepaths for OG images
        2. Save 8 images for inference
        3. Crop remaining images into patches
        4. Split into train/val/test
        5. Augment some of the training images
        """
        print("="*70)
        print(f"PREPROCESSING PIPELINE")
        print("="*70)

        inference_dir = Path(self.config.paths.inference_dir)
        cropped_dir = Path(self.config.paths.cropped_img_root)

        # Check if inference image dir exists and is not empty
        if inference_dir.exists() and inference_dir.is_dir() and any(inference_dir.iterdir()):
            print(f"Inference images already saved in {inference_dir}.\nSkipping steps 1, 2 and 3.")
        else:
            # 1. Get filepaths for all OG images
            print(f"\n1. Getting filepaths for original images...")
            non_det_fp_list, gt_fp_list, noisy_fp_list = read_in_og_images(
                top_dir=self.config.paths.og_images_root
            )

            # 2. Save 8 images for inference
            print(f"\n2. Saving inference images...")
            num_inf_imgs = self.config.dataset.num_inference_imgs
            shuffled_gt_list, shuffled_noisy_list = shuffle(gt_fp_list, noisy_fp_list)
            inf_gt_list = shuffled_gt_list[:num_inf_imgs]
            shuffled_gt_list = shuffled_gt_list[num_inf_imgs:]
            inf_noisy_list = shuffled_noisy_list[:num_inf_imgs]
            shuffled_noisy_list = shuffled_noisy_list[num_inf_imgs:]

            save_inference(
                inf_gt_list=inf_gt_list,
                inf_noisy_list=inf_noisy_list,
                save_dir=inference_dir
            )

            # Check if crop image dirs exist and are not empty
            if cropped_dir.exists() and cropped_dir.is_dir() and any(cropped_dir.iterdir()):
                print(f"Image crops already saved in {cropped_dir}. Skipping step 3.")
            else:
                # 3. Crop images
                print(f"\n3. Cropping remaining images into patches...")
                segmentation_process(
                    gt_fp_list=shuffled_gt_list,
                    noisy_fp_list=shuffled_noisy_list,
                    save_as_root=cropped_dir,
                    patch_size=self.config.dataset.patch_size
                )

        # 4. Split dataset
        print(f"\n4. Splitting into train/val/test...")
        (self.gt_train_list, self.noisy_train_list,
         self.gt_val_list, self.noisy_val_list,
         self.gt_test_list, self.noisy_test_list) = prepare_and_split(
            config=self.config
        )

        # 5. Augment training images
        print(f"\n5. Augmenting training images...")
        self.gt_train_list, self.noisy_train_list = augment_images(
            og_gt_img_names=self.gt_train_list,
            og_noisy_img_names=self.noisy_train_list,
            crop_img_root=cropped_dir,
            bright_threshold=self.config.preprocessing.bright_threshold,
            bright_copies=self.config.preprocessing.bright_copies,        # Copies per bright image
            random_n=self.config.preprocessing.random_augment             # Number of random non-bright images
        )

        print(f"\n  Preprocessing complete!")
        print(f"    Training pairs: {len(self.gt_train_list)}")
        print(f"    Validation pairs: {len(self.gt_val_list)}")
        print(f"    Test pairs: {len(self.gt_test_list)}")

        # 6. Save test split for reproducibility
        print(f"\n6. Saving test split...")

        test_split = {
            "gt_test_list": self.gt_test_list,
            "noisy_test_list": self.noisy_test_list,
            "gt_val_list": self.gt_val_list,
            "noisy_val_list": self.noisy_val_list
        }

        split_file = cropped_dir / "data_split.json"
        with open(split_file, "w") as f:
            json.dump(test_split, f, indent=2)

        print(f"    Data split saved to {split_file}")

    def create_dataloaders(
            self,
            train_size: Optional[int] = None,
            val_size: Optional[int] = None,
            test_size: Optional[int] = None
    ) -> None:
        """
        Create train, validation and test dataloaders.

        Args:
            train_size: Number of training pairs to use. If None, uses all
            val_size: Number of validation pairs to use. If None, uses all
            test_size: Number of test pairs to use. If None, uses all.
        """
        print("\n" + "="*70)
        print("CREATING DATALOADERS")
        print("="*70)

        # Determine sizes
        train_size = train_size or len(self.gt_train_list)
        val_size = val_size or len(self.gt_val_list)
        test_size = test_size or len(self.gt_test_list)

        # Create dataloaders
        self.train_loader = get_dataloader(
            config=self.config,
            gt_img_names=self.gt_train_list[:train_size],
            noisy_img_names=self.noisy_train_list[:train_size],
            device=self.device,
            batch_size=self.config.test.batch_size,
            collate_fn=collate,
            augment=self.config.augmentation.enabled
        )

        self.val_loader = get_dataloader(
            config=self.config,
            gt_img_names=self.gt_val_list[:val_size],
            noisy_img_names=self.noisy_val_list[:val_size],
            device=self.device,
            batch_size=self.config.train.batch_size,
            collate_fn=collate,
            shuffle=False,
            augment=False
        )

        self.test_loader = get_dataloader(
            config=self.config,
            gt_img_names=self.gt_test_list[:test_size],
            noisy_img_names=self.noisy_test_list[:test_size],
            device=self.device,
            batch_size=self.config.train.batch_size,
            collate_fn=collate,
            shuffle=False,
            augment=False
        )

        print(f"Created train_loader: {len(self.train_loader.dataset)} pairs, {len(self.train_loader)} batches")
        print(f"Created val_loader: {len(self.val_loader.dataset)} pairs, {len(self.val_loader)} batches")
        print(f"Created test_loader: {len(self.test_loader.dataset)} pairs, {len(self.test_loader)} batches")

    def setup_model(self) -> None:
        """
        Setup model, metrics and validation function.

        Creates U-Net model, PSNR/SSIM metrics and validation function.
        """
        print("\n" + "="*70)
        print("SETTING UP MODEL")
        print("="*70)

        # Create model
        self.model = get_model(device=self.device, config=self.config)
        print(f"Model created")

        # Create metrics
        self.psnr_metric, self.ssim_metric = create_metrics_objects(
            device=self.device
        )
        print(f"Metrics created")

        # Create validation function
        self.val_function = validate_function(
            val_loader=self.val_loader,
            config=self.config,
            device=self.device,
            psnr_metric=self.psnr_metric,
            ssim_metric=self.ssim_metric
        )
        print(f"Validation function created")

    def run_training(
            self,
            resume_path: Optional[str | Path] = None
    ) -> None:
        """
        Run model training.

        Trains model with validation, checkpointing and metric tracking.
        Updates self.history with training metrics and self.checkpoint_dir
        with the run directory.

        Args:
            resume_path: Path of checkpoint file to resume from. If None,
                uses config.train.resume_from or trains from scratch.

        """
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)

        # Use config resume path if not specified
        if resume_path is None:
            resume_path = self.config.train.resume_from

        # Train
        self.history, self.checkpoint_dir = train(
            model=self.model,
            train_loader=self.train_loader,
            config=self.config,
            device=self.device,
            val_fn=self.val_function,
            val_loader=self.val_loader,
            resume_path=resume_path,
            psnr_metric=self.psnr_metric,
            ssim_metric=self.ssim_metric
        )

        print(f"\n" + "="*70)
        print(f"TRAINING COMPLETE")
        print("="*70)
        print(f"Results saved to {self.checkpoint_dir}")

        # Save training history
        save_training_history(
            history=self.history,
            save_path=self.checkpoint_dir / "training_history.json"
        )
        print(f"Training history saved")

    def get_summary(self) -> None:
        """
        Print summary of pipeline state.
        """
        print("\n" + "="*70)
        print(f"PIPELINE SUMMARY")
        print("="*70)

        print(f"Device: {self.device}")
        print(f"Config: {self.config}")

        if self.train_loader:
            print(f"\nDataloaders:")
            print(f"    Train: {len(self.train_loader.dataset)} pairs")
            print(f"    Val: {len(self.val_loader.dataset)} pairs")
            print(f"    Test: {len(self.test_loader.dataset)} pairs")

        if self.history:
            print(f"\nTraining:")
            print(f"    Epochs: {len(self.history['epoch'])}")
            print(f"    Best val PSNR: {max([x for x in self.history['val_psnr'] if x is not None]):.2f} dB")
            print(f"    Best val SSIM: {max([x for x in self.history['val_ssim'] if x is not None]):.4f} ")

        if self.checkpoint_dir:
            print(f"\nCheckpoint: {self.checkpoint_dir}")

        print("="*70)

def run_training_pipeline(
        config: Config,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None
) -> TrainingPipeline:
    """
    Run complete training pipeline in one function call.

    Executes preprocessing, dataloader creation, model setup and training.
    Automatically saves checkpoints and training history.

    Args:
        config: Configuration object
        train_size: Number of training pairs. If None, uses all available
        val_size: Number of validation pairs. If None, uses all available
        test_size: Number of test pairs. If None, uses all available

    Returns:
         Pipeline object with trained model and training history
    """
    # Create pipeline
    pipeline = TrainingPipeline(config=config)

    # Run all steps
    pipeline.run_preprocessing()
    pipeline.create_dataloaders(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size
    )
    pipeline.setup_model()
    pipeline.run_training()

    # Print summary
    pipeline.get_summary()

    return pipeline




