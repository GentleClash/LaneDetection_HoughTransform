import os
import cv2
import numpy as np
from typing import Tuple, List
import random
import matplotlib.pyplot as plt

class MSRParameterOptimizer:
    """
    Self-learning optimizer to find best alpha/beta values for MSR color restoration
    Supports multiple images with varying conditions
    """
    
    def __init__(self, images: List[np.ndarray] = None, image: np.ndarray = None):
        # Handle both single image and multiple images
        if images is not None:
            self.images = [img.copy() for img in images]
        elif image is not None:
            self.images = [image.copy()]
        else:
            raise ValueError("Must provide either 'images' list or single 'image'")
        
        # Pre-compute reference edges for all images
        self.reference_edges = [self._get_reference_edges(img) for img in self.images]
        
        # RL parameters
        self.learning_rate = 0.1
        self.exploration_rate = 0.3
        self.exploration_decay = 0.95
        self.min_exploration = 0.05
        
        # Parameter search space
        self.alpha_range = (10, 200)
        self.beta_range = (0.5, 50)
        
        # Experience replay
        self.memory = []
        self.memory_size = 100
        
        # Best parameters tracking
        self.best_score = -np.inf
        self.best_params = (125, 1)
        
        # MSR fixed parameters
        self.msr_scales = [15, 50]
        self.msr_weights = [0.5, 0.5]
    
    def _get_reference_edges(self, image: np.ndarray) -> np.ndarray:
        """Get Canny edges from original image as reference"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply median blur to match the processing pipeline
        blurred = cv2.medianBlur(gray, 5)
        edges = cv2.Canny(blurred, 50, 150)
        return edges
    def _calculate_multi_image_reward(self, alpha: float, beta: float) -> float:
        """
        Calculate average reward across all images for robustness
        """
        total_reward = 0
        valid_images = 0

        for i, (image, ref_edges) in enumerate(zip(self.images, self.reference_edges)):
            try:
                processed_image = self._apply_msr_with_params(image, alpha, beta)
                reward = self._calculate_reward(processed_image, ref_edges)
                total_reward += reward
                valid_images += 1
            except Exception as e:
                print(f"Warning: Error processing image {i}: {e}")
                continue

        if valid_images == 0:
            return -1.0

        # Average reward across all valid images
        avg_reward = total_reward / valid_images

        # Bonus for consistency (lower variance across images)
        if len(self.images) > 1:
            image_rewards = []
            for image, ref_edges in zip(self.images, self.reference_edges):
                try:
                    processed_image = self._apply_msr_with_params(image, alpha, beta)
                    reward = self._calculate_reward(processed_image, ref_edges)
                    image_rewards.append(reward)
                except:
                    continue

            if len(image_rewards) > 1:
                consistency_bonus = 0.1 * (1 - np.std(image_rewards))  # Reward low variance
                avg_reward += consistency_bonus

        return avg_reward
    
    def _apply_msr_with_params(self, image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """Apply MSR with specific alpha/beta parameters to a given image"""
        
        if len(image.shape) != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Convert to float and normalize
        img_float = image.astype(np.float32) / 255.0
        img_float = np.maximum(img_float, 1e-6)
        
        # Apply MSR to each channel
        msr_result = np.zeros_like(img_float)
        
        for channel in range(3):
            channel_data = img_float[:, :, channel]
            log_img = np.log(channel_data)
            
            msr_channel = np.zeros_like(channel_data)
            
            for scale, weight in zip(self.msr_scales, self.msr_weights):
                kernel_size = min(int(3 * scale + 1), 31)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                    
                kernel = cv2.getGaussianKernel(kernel_size, scale)
                kernel_2d = np.outer(kernel, kernel)
                kernel_2d = kernel_2d / np.sum(kernel_2d)
                
                convolved = cv2.filter2D(channel_data, -1, kernel_2d, borderType=cv2.BORDER_REFLECT)
                convolved = np.maximum(convolved, 1e-6)
                
                ssr = log_img - np.log(convolved)
                msr_channel += weight * ssr
            
            msr_result[:, :, channel] = msr_channel
        
        # Color Restoration with custom alpha/beta
        sum_channels = np.sum(img_float, axis=2, keepdims=True)
        sum_channels = np.maximum(sum_channels, 1e-6)
        
        color_restoration = np.zeros_like(img_float)
        for channel in range(3):
            ratio = img_float[:, :, channel] / sum_channels[:, :, 0]
            color_restoration[:, :, channel] = beta * np.log(alpha * ratio + 1e-6)
        
        # Apply color restoration
        msrcr_result = color_restoration * msr_result
        
        # Normalize
        msrcr_result = np.clip(msrcr_result, -3, 3)
        msrcr_result = (msrcr_result - np.min(msrcr_result)) / (np.max(msrcr_result) - np.min(msrcr_result))
        msrcr_result = (msrcr_result * 255).astype(np.uint8)
        
        return msrcr_result
    
    def _calculate_reward(self, processed_image: np.ndarray, reference_edges: np.ndarray) -> float:
        """
        Calculate reward based on edge similarity to reference
        Higher reward for edges closer to reference but not identical
        """
        # Apply median blur as specified
        gray_processed = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        blurred_processed = cv2.medianBlur(gray_processed, 5)
        
        # Get Canny edges
        processed_edges = cv2.Canny(blurred_processed, 50, 150)
        
        # Calculate similarity metrics
        # 1. Structural Similarity (SSIM-like)
        diff = np.abs(processed_edges.astype(float) - reference_edges.astype(float))
        mse = np.mean(diff ** 2)
        structural_similarity = 1 / (1 + mse / 100.0)  # Normalized MSE
        
        # 2. Edge density similarity
        orig_edge_density = np.sum(reference_edges > 0) / reference_edges.size
        proc_edge_density = np.sum(processed_edges > 0) / processed_edges.size
        density_similarity = 1 - abs(orig_edge_density - proc_edge_density)
        
        # 3. Penalize identical images (we want enhancement, not preservation)
        identical_penalty = 0
        if np.array_equal(processed_edges, reference_edges):
            identical_penalty = -0.5
        
        # 4. Image quality metrics
        # Penalize extreme values or artifacts
        quality_penalty = 0
        if np.std(processed_image) < 20:  # Too flat
            quality_penalty -= 0.3
        elif np.std(processed_image) > 80:  # Too noisy
            quality_penalty -= 0.3
        
        # Combined reward
        reward = (0.5 * structural_similarity + 
                 0.3 * density_similarity + 
                 0.2 * quality_penalty + 
                 identical_penalty)
        
        return reward
    
    def _epsilon_greedy_action(self) -> Tuple[float, float]:
        """Choose action using epsilon-greedy strategy"""
        if random.random() < self.exploration_rate:
            # Explore: random parameters
            alpha = random.uniform(*self.alpha_range)
            beta = random.uniform(*self.beta_range)
        else:
            # Exploit: use best known parameters with small perturbation
            alpha = self.best_params[0] + random.gauss(0, 10)
            beta = self.best_params[1] + random.gauss(0, 2)
            
            # Clip to valid ranges
            alpha = np.clip(alpha, *self.alpha_range)
            beta = np.clip(beta, *self.beta_range)
        
        return alpha, beta
    
    def _update_memory(self, alpha: float, beta: float, reward: float):
        """Store experience in replay memory"""
        experience = (alpha, beta, reward)
        
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)  # Remove oldest
        
        self.memory.append(experience)
    
    def _learn_from_memory(self):
        """Learn from past experiences using weighted sampling"""
        if len(self.memory) < 10:
            return
        
        # Sample experiences weighted by reward
        experiences = np.array(self.memory)
        rewards = experiences[:, 2]
        
        # Normalize rewards for probability weights
        if np.std(rewards) > 0:
            normalized_rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))
            weights = normalized_rewards + 0.1  # Avoid zero weights
            weights = weights / np.sum(weights)
            
            # Sample good experiences
            sample_indices = np.random.choice(len(self.memory), size=min(5, len(self.memory)), 
                                            replace=False, p=weights)
            
            # Update best parameters based on weighted average of good experiences
            good_experiences = experiences[sample_indices]
            good_alphas = good_experiences[:, 0]
            good_betas = good_experiences[:, 1]
            good_rewards = good_experiences[:, 2]
            
            # Weighted average towards better parameters
            weight_sum = np.sum(good_rewards)
            if weight_sum > 0:
                avg_alpha = np.average(good_alphas, weights=good_rewards)
                avg_beta = np.average(good_betas, weights=good_rewards)
                
                # Move best params towards weighted average
                self.best_params = (
                    self.best_params[0] * (1 - self.learning_rate) + avg_alpha * self.learning_rate,
                    self.best_params[1] * (1 - self.learning_rate) + avg_beta * self.learning_rate
                )
    
    def optimize(self, episodes: int = 50) -> Tuple[float, float]:
        """
        Run optimization for specified episodes on multiple images
        Returns best (alpha, beta) parameters found
        """
        scores = []
        best_scores = []
        alpha_history = []
        beta_history = []
        
        print(f"Starting optimization with {episodes} episodes on {len(self.images)} images...")
        print(f"Initial parameters: alpha={self.best_params[0]:.1f}, beta={self.best_params[1]:.1f}")
        
        for episode in range(episodes):
            # Choose action
            alpha, beta = self._epsilon_greedy_action()
            
            try:
                # Calculate reward across all images
                reward = self._calculate_multi_image_reward(alpha, beta)
                
                # Update memory
                self._update_memory(alpha, beta, reward)
                
                # Track parameter history
                alpha_history.append(alpha)
                beta_history.append(beta)
                
                # Update best parameters if better reward found
                if reward > self.best_score:
                    self.best_score = reward
                    self.best_params = (alpha, beta)
                    print(f"Episode {episode}: New best! α={alpha:.1f}, β={beta:.1f}, reward={reward:.3f}")
                
                # Learn from memory
                if episode % 5 == 0:
                    self._learn_from_memory()
                
                # Decay exploration
                self.exploration_rate = max(self.min_exploration, 
                                          self.exploration_rate * self.exploration_decay)
                
                scores.append(reward)
                best_scores.append(self.best_score)
                
                if episode % 10 == 0:
                    print(f"Episode {episode}: α={alpha:.1f}, β={beta:.1f}, reward={reward:.3f}, "
                          f"exploration={self.exploration_rate:.3f}")
                    
            except Exception as e:
                print(f"Episode {episode}: Error with α={alpha:.1f}, β={beta:.1f}: {e}")
                scores.append(-1.0)
                best_scores.append(self.best_score)
                alpha_history.append(alpha)
                beta_history.append(beta)
        
        print(f"\nOptimization complete!")
        print(f"Best parameters: α={self.best_params[0]:.1f}, β={self.best_params[1]:.1f}")
        print(f"Best score: {self.best_score:.3f}")
        
        # Plot optimization progress
        self._plot_optimization_progress(scores, best_scores, alpha_history, beta_history)
        
        return self.best_params
    
    def _plot_optimization_progress(self, scores: List[float], best_scores: List[float], 
                                   alpha_history: List[float], beta_history: List[float]):
        """Plot optimization progress graphs"""
        episodes = range(len(scores))
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MSR Parameter Optimization Progress', fontsize=16)
        
        # Plot 1: Reward progression
        ax1.plot(episodes, scores, alpha=0.7, label='Episode Reward', color='blue')
        ax1.plot(episodes, best_scores, label='Best Score', color='red', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Reward Progression')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Alpha parameter evolution
        ax2.plot(episodes, alpha_history, alpha=0.7, color='green')
        ax2.axhline(y=self.best_params[0], color='red', linestyle='--', label=f'Best α={self.best_params[0]:.1f}')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Alpha')
        ax2.set_title('Alpha Parameter Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Beta parameter evolution
        ax3.plot(episodes, beta_history, alpha=0.7, color='orange')
        ax3.axhline(y=self.best_params[1], color='red', linestyle='--', label=f'Best β={self.best_params[1]:.1f}')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Beta')
        ax3.set_title('Beta Parameter Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Parameter space exploration (scatter plot)
        scatter = ax4.scatter(alpha_history, beta_history, c=scores, cmap='viridis', alpha=0.6)
        ax4.scatter(self.best_params[0], self.best_params[1], color='red', s=100, marker='*', 
                   label=f'Best (α={self.best_params[0]:.1f}, β={self.best_params[1]:.1f})')
        ax4.set_xlabel('Alpha')
        ax4.set_ylabel('Beta')
        ax4.set_title('Parameter Space Exploration')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Reward')
        
        plt.tight_layout()
        plt.savefig('msr_optimization_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Optimization progress graphs saved as 'msr_optimization_progress.png'")
    
    def visualize_results(self, save_all: bool = True):
        """Show comparison of original vs optimized MSR for all images"""
        print(f"Generating visualizations for {len(self.images)} images...")
        
        for i, (image, ref_edges) in enumerate(zip(self.images, self.reference_edges)):
            # Original edges
            orig_edges = ref_edges
            
            # Optimized MSR
            optimized_msr = self._apply_msr_with_params(image, *self.best_params)
            gray_opt = cv2.cvtColor(optimized_msr, cv2.COLOR_BGR2GRAY)
            blurred_opt = cv2.medianBlur(gray_opt, 5)
            opt_edges = cv2.Canny(blurred_opt, 50, 150)
            
            # Create directories if they don't exist
            os.makedirs('msr_results', exist_ok=True)
            os.makedirs('edges_results', exist_ok=True)
            
            if save_all or i == 0:  # Save all or just first image
                # Create side-by-side comparison for MSR images
                h, w, c = image.shape
                msr_comparison = np.zeros((h, w * 2, c), dtype=np.uint8)
                
                # Place original image on left, optimized on right
                msr_comparison[:, :w] = image
                msr_comparison[:, w:] = optimized_msr
                
                # Add text labels for MSR comparison
                cv2.putText(msr_comparison, 'Original', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(msr_comparison, f'MSR (a={self.best_params[0]:.1f}, b={self.best_params[1]:.1f})', 
                           (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Create side-by-side comparison for Canny edges
                h_edge, w_edge = orig_edges.shape
                edge_comparison = np.zeros((h_edge, w_edge * 2), dtype=np.uint8)
                
                # Place original edges on left, optimized on right
                edge_comparison[:, :w_edge] = orig_edges
                edge_comparison[:, w_edge:] = opt_edges
                
                # Add text labels for edge comparison
                edge_comparison_colored = cv2.cvtColor(edge_comparison, cv2.COLOR_GRAY2BGR)
                cv2.putText(edge_comparison_colored, 'Original Edges', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(edge_comparison_colored, 'Optimized Edges', (w_edge + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Save in organized folders
                cv2.imwrite(f'msr_results/image_{i:02d}_comparison.jpg', msr_comparison)
                cv2.imwrite(f'edges_results/image_{i:02d}_edges_comparison.jpg', edge_comparison_colored)
        
        if save_all:
            print(f"Saved visualization files for all {len(self.images)} images")
        else:
            print("Saved visualization files for first image only")
            
    def get_performance_report(self) -> dict:
        """Generate detailed performance report across all images"""
        report = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'num_images': len(self.images),
            'individual_scores': [],
            'score_stats': {}
        }
        
        # Calculate individual scores for each image
        for i, (image, ref_edges) in enumerate(zip(self.images, self.reference_edges)):
            try:
                processed_image = self._apply_msr_with_params(image, *self.best_params)
                score = self._calculate_reward(processed_image, ref_edges)
                report['individual_scores'].append({
                    'image_index': i,
                    'score': score
                })
            except Exception as e:
                report['individual_scores'].append({
                    'image_index': i,
                    'score': None,
                    'error': str(e)
                })
        
        # Calculate statistics
        valid_scores = [item['score'] for item in report['individual_scores'] if item['score'] is not None]
        if valid_scores:
            report['score_stats'] = {
                'mean': np.mean(valid_scores),
                'std': np.std(valid_scores),
                'min': np.min(valid_scores),
                'max': np.max(valid_scores),
                'median': np.median(valid_scores)
            }
        
        return report

# Example usage:
if __name__ == "__main__":
    from pathlib import Path 
    # Method 1: Load multiple images from a list of paths
    #image_paths = ['test_image.jpg', '1.webp']
    #images = []
    #for path in image_paths:
    #    img = cv2.imread(path)
    #    if img is not None:
    #        images.append(img)
    #        print(f"Loaded: {path}")
    #    else:
    #
    #print(f"Warning: Could not load {path}")
    
    
    image_dir = "tmp/"  
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

    images = []
    for file_path in Path(image_dir).iterdir():
        if file_path.suffix.lower() in image_extensions:
            img = cv2.imread(str(file_path))
            if img is not None:
                images.append(img)
                print(f"Loaded: {file_path.name}")

    if len(images) == 0:
        print("No images loaded. Please check file paths.")
        exit()

    # Create optimizer with multiple images
    print(f"\nOptimizing MSR parameters for {len(images)} images...")
    optimizer = MSRParameterOptimizer(images=images)

    # Run optimization (more episodes for multiple images)
    episodes = 50 if len(images) > 3 else 30
    best_alpha, best_beta = optimizer.optimize(episodes=episodes)

    # Generate detailed performance report
    report = optimizer.get_performance_report()

    print(f"\n=== OPTIMIZATION REPORT ===")
    print(f"Best parameters: α={best_alpha:.1f}, β={best_beta:.1f}")
    print(f"Overall best score: {report['best_score']:.3f}")
    print(f"Images processed: {report['num_images']}")

    if 'score_stats' in report and report['score_stats']:
        stats = report['score_stats']
        print(f"\nScore Statistics:")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Std:  {stats['std']:.3f}")
        print(f"  Min:  {stats['min']:.3f}")
        print(f"  Max:  {stats['max']:.3f}")

    print(f"\nIndividual image scores:")
    for item in report['individual_scores']:
        if item['score'] is not None:
            print(f"  Image {item['image_index']:2d}: {item['score']:.3f}")
        else:
            print(f"  Image {item['image_index']:2d}: ERROR - {item.get('error', 'Unknown error')}")

    # Visualize results (save all comparison images)
    optimizer.visualize_results(save_all=True)

    print(f"self.color_restore_alpha = {best_alpha:.1f}")
    print(f"self.color_restore_beta = {best_beta:.1f}")
    
    # Method 2: Alternative - Load images from a directory
    """
    import os
    from pathlib import Path
    
    image_dir = "test_images/"  # Directory withtest images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    images = []
    for file_path in Path(image_dir).iterdir():
        if file_path.suffix.lower() in image_extensions:
            img = cv2.imread(str(file_path))
            if img is not None:
                images.append(img)
                print(f"Loaded: {file_path.name}")
    
    if images:
        optimizer = MSRParameterOptimizer(images=images)
        best_alpha, best_beta = optimizer.optimize(episodes=40)
        optimizer.visualize_results()
    """
