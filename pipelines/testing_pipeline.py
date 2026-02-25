from utils import *
from imports import *
from models import *
from configs import Config
from typing import Optional

class TestingPipeline:
    """
    Testing pipeline for model evaluation.

    Loads trained model, runs evaluation on test set and saves results.
    """
    def __init__(
            self,
            config: Config,
            test_loader: DataLoader,
            checkpoint_dir: Optional[str | Path] = None
    ) -> None:
        """
        Initialise testing pipeline.

        Args:
            config: Configuration object
            test_loader: Pre-created test DataLoader
            checkpoint_dir: Directory with trained model
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialise attributes
        self.test_loader = test_loader
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.model = None
        self.psnr_metric = None
        self.ssim_metric = None
        self.test_results = None

        print(f"Using device: {self.device}")

    def load_checkpoint(
            self,
            checkpoint_dir: str | Path
    ) -> None:
        """
        Load trained model from checkpoint directory.

        Args:
            checkpoint_dir: Directory containing best_model.pth
        """
        print("\n" + "="*70)
        print("LOADING CHECKPOINT")
        print("="*70)

        self.checkpoint_dir = Path(checkpoint_dir)
        checkpoint_path = self.checkpoint_dir / "best_model.pth"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")

        # Create model and metrics
        self.model = get_model(device=self.device, config=self.config)
        self.psnr_metric, self.ssim_metric = create_metrics_objects(device=self.device)

        # Load checkpoint
        self.model = load_checkpoint_inference(
            checkpoint_path=checkpoint_path,
            model=self.model,
            device=self.device
        )

        print(f"Loaded model from {checkpoint_path}")

    def run_testing(self) -> None:
        """
        Run testing on test set.

        Evaluates model on test data and stores results in self.test_results.
        """
        print("\n" + "="*70)
        print("TESTING")
        print("="*70)

        if self.model is None:
            raise ValueError("Model not loaded. Call load_checkpoint() first.")

        if self.test_loader is None:
            raise ValueError("Test loader not loaded.")

        # Run test
        self.test_results = test(
            model=self.model,
            test_loader=self.test_loader,
            device=self.device,
            psnr_metric=self.psnr_metric,
            ssim_metric=self.ssim_metric,
            save_dir=self.checkpoint_dir / "test_results" if self.checkpoint_dir else None
        )

        print("\n" + "="*70)
        print("TESTING COMPLETE")
        print("="*70)
        print_results(self.test_results)

    def save_results(self) -> None:
        """
        Save test results and visualisations.

        Saves to checkpoint directory:
        - Test results JSON
        - Summary TXT (if training history available)
        - All visualisations
        """
        print("\n" + "="*70)
        print("SAVING RESULTS & VISUALISATIONS")
        print("="*70)

        if self.test_results is None:
            raise ValueError("No test results. Run run_testing() first.")

        if self.checkpoint_dir is None:
            raise ValueError("No checkpoint directory specified.")

        # Save test results
        save_test_results(
            results=self.test_results,
            save_path=self.checkpoint_dir / "test_results.json"
        )
        print(f"Test results saved")

        # Save summary and all visualisations (if training history exists)
        history_path = self.checkpoint_dir / "training_history.json"
        if history_path.exists():
            history = load_training_history(load_path=history_path)
            save_summary(
                history=history,
                results=self.test_results,
                save_path=self.checkpoint_dir / "summary.txt"
            )
            print("Summary saved")

            save_all_visualisations(
                history=history,
                results=self.test_results,
                test_loader=self.test_loader,
                model=self.model,
                device=self.device,
                checkpoint_dir=self.checkpoint_dir
            )
        else:
            # Just save test visualisations
            figures_dir = self.checkpoint_dir / "figures"
            figures_dir.mkdir(exist_ok=True)

            plot_test_results_distribution(
                results=self.test_results,
                save_path=figures_dir / "test_results_distribution.png"
            )
            plot_metric_vs_metric(
                results=self.test_results,
                save_path=figures_dir / "psnr_vs_ssim.png"
            )
            show_best_worst_results(
                results=self.test_results,
                test_loader=self.test_loader,
                model=self.model,
                device=self.device,
                save_path=figures_dir / "best_worst_results.png"
            )
        print("All visualisations saved")

    def run_testing_workflow(self) -> None:
        """
        Run complete testing workflow: test -> save results.
        """
        self.run_testing()
        self.save_results()

    def get_summary(self) -> None:
        """
        Print summary of test results.
        """
        print("\n" + "="*70)
        print("TESTING SUMMARY")
        print("="*70)

        if self.test_results:
            print(f"\nTest Results:")
            print(f"    Images tested: {self.test_results['num_images']}")
            print(f"    Avg PSNR: {self.test_results['avg_psnr']:.2f} dB")
            print(f"    Avg SSIM: {self.test_results['avg_ssim']:.4f} ")

        if self.checkpoint_dir:
            print(f"\nResults saved to: {self.checkpoint_dir}")

        print("="*70)

def run_testing_pipeline(
        config: Config,
        checkpoint_dir: str | Path,
        test_loader: DataLoader = None,
) -> TestingPipeline:
    """
    Run complete testing pipeline in one function call.

    Args:
        config: Configuration object
        checkpoint_dir: Directory containing trained model checkpoint
        test_loader: Pre-created test DataLoader

    Returns:
        TestingPipeline object with test results
    """
    pipeline = TestingPipeline(config=config, checkpoint_dir=checkpoint_dir, test_loader=test_loader)
    pipeline.load_checkpoint(checkpoint_dir=checkpoint_dir)
    pipeline.run_testing_workflow()
    pipeline.get_summary()

    return pipeline
