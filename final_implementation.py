import cv2
import numpy as np
import time
import random
from sklearn.cluster import DBSCAN
from typing import Tuple, List, Optional

class GeneticThresholdOptimizer:
    """
    GENETIC ALGORITHM FOR AUTOMATIC THRESHOLD OPTIMIZATION
    """

    def __init__(self, population_size=30, generations=15, mutation_rate=0.1) -> None:
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = 0.8

        # Threshold ranges optimized for lane detection
        self.min_threshold = 20
        self.max_threshold = 200

        # Best solution tracking
        self.best_fitness_history = []
        self.best_solution = None
        self.best_fitness = float('-inf')

    def create_individual(self) -> Tuple[int, int]:
        """Create a random individual representing threshold pair"""
        low_threshold = random.randint(self.min_threshold, self.max_threshold // 2)
        high_threshold = random.randint(low_threshold + 10, self.max_threshold)
        return (low_threshold, high_threshold)

    def initialize_population(self) -> List[Tuple[int, int]]:
        """Initialize population of threshold pairs"""
        return [self.create_individual() for _ in range(self.population_size)]

    def fitness_function(self, individual: Tuple[int, int], image: np.ndarray) -> float:
        """
        FITNESS FUNCTION - Optimized for lane detection
        Evaluates threshold quality based on lane detection effectiveness
        """
        if image is None or image.size == 0:
            return -1000

        low_thresh, high_thresh = individual

        if low_thresh >= high_thresh or low_thresh < 10:
            return -1000

        try:
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply region of interest (simplified version)
            height, width = gray.shape
            vertices = np.array([
                [(width * 0.1, height),
                 (width * 0.4, height * 0.6),
                 (width * 0.6, height * 0.6),
                 (width * 0.9, height)]
            ], dtype=np.int32)

            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [vertices[0]], (255,))
            roi_gray = cv2.bitwise_and(blurred, mask)

            # Apply Canny with these thresholds
            edges = cv2.Canny(roi_gray, low_thresh, high_thresh)

            # Calculate fitness components for lane detection
            edge_density = self._calculate_lane_edge_density(edges)
            connectivity = self._calculate_lane_connectivity(edges)
            line_quality = self._calculate_line_quality(edges)

            # Combined fitness for lane detection
            fitness = (
                0.4 * edge_density +      # 40% weight on appropriate edge density
                0.4 * connectivity +      # 40% weight on edge connectivity
                0.2 * line_quality        # 20% weight on line-like structures
            )

            return fitness

        except Exception as e:
            print(f"⚠️  Error in fitness function: {e}")
            return -1000

    def _calculate_lane_edge_density(self, edges: np.ndarray) -> float:
        """Calculate appropriate edge density for lane detection"""
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_pixels = np.sum(edges > 0)
        density = edge_pixels / total_pixels

        # Optimal density for lane detection is around 2-8%
        optimal_density = 0.05
        penalty = abs(density - optimal_density) / optimal_density
        return max(0, 100 * (1 - penalty))

    def _calculate_lane_connectivity(self, edges: np.ndarray) -> float:
        """Calculate connectivity suitable for lane lines"""
        if np.sum(edges) == 0:
            return 0

        # Use morphological operations to find connected components
        kernel = np.ones((3, 3), np.uint8)
        connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        num_labels, labels = cv2.connectedComponents(connected)

        if num_labels <= 1:
            return 0

        # Prefer fewer, larger components (lane-like structures)
        component_sizes = []
        for i in range(1, num_labels):
            component_size = np.sum(labels == i)
            component_sizes.append(component_size)

        if not component_sizes:
            return 0

        # Reward larger components
        avg_component_size = float(np.mean(component_sizes))
        return min(avg_component_size / 50.0, 100.0)

    def _calculate_line_quality(self, edges: np.ndarray) -> float:
        """Calculate how well edges form line-like structures"""
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                               minLineLength=30, maxLineGap=10)

        if lines is None:
            return 0

        # Reward having some lines, but not too many (noise)
        num_lines = len(lines)
        if num_lines == 0:
            return 0
        elif num_lines <= 20:
            return min(num_lines * 5, 100)
        else:
            return max(0, 100 - (num_lines - 20) * 2)

    def selection(self, population: List[Tuple[int, int]],
                  fitness_scores: List[float]) -> List[Tuple[int, int]]:
        """Tournament selection"""
        selected = []
        tournament_size = 3

        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_index])

        return selected

    def crossover(self, parent1: Tuple[int, int], parent2: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Crossover to create offspring"""
        if random.random() > self.crossover_rate:
            return parent1, parent2

        p1_low, p1_high = parent1
        p2_low, p2_high = parent2

        alpha = random.random()
        offspring1_low = int(alpha * p1_low + (1 - alpha) * p2_low)
        offspring1_high = int(alpha * p1_high + (1 - alpha) * p2_high)

        offspring2_low = int((1 - alpha) * p1_low + alpha * p2_low)
        offspring2_high = int((1 - alpha) * p1_high + alpha * p2_high)

        # Ensure valid threshold pairs
        offspring1_low = max(self.min_threshold, min(offspring1_low, offspring1_high - 10))
        offspring1_high = max(offspring1_low + 10, min(offspring1_high, self.max_threshold))

        offspring2_low = max(self.min_threshold, min(offspring2_low, offspring2_high - 10))
        offspring2_high = max(offspring2_low + 10, min(offspring2_high, self.max_threshold))

        return (offspring1_low, offspring1_high), (offspring2_low, offspring2_high)

    def mutate(self, individual: Tuple[int, int]) -> Tuple[int, int]:
        """Mutation operator"""
        if random.random() > self.mutation_rate:
            return individual

        low_thresh, high_thresh = individual

        if random.random() < 0.5:
            mutation_range = 15
            low_thresh += random.randint(-mutation_range, mutation_range)
            low_thresh = max(self.min_threshold, min(low_thresh, high_thresh - 10))
        else:
            mutation_range = 15
            high_thresh += random.randint(-mutation_range, mutation_range)
            high_thresh = max(low_thresh + 10, min(high_thresh, self.max_threshold))

        return (low_thresh, high_thresh)

    def evolve(self, image: np.ndarray, verbose: bool = False) -> Tuple[int, int]:
        """Main evolution process"""
        if verbose:
            print("Optimizing Canny thresholds with Genetic Algorithm...")

        population = self.initialize_population()

        for generation in range(self.generations):
            fitness_scores = [self.fitness_function(ind, image) for ind in population]

            best_idx = np.argmax(fitness_scores)
            generation_best_fitness = fitness_scores[best_idx]
            generation_best_individual = population[best_idx]

            if generation_best_fitness > self.best_fitness:
                self.best_fitness = generation_best_fitness
                self.best_solution = generation_best_individual

            self.best_fitness_history.append(self.best_fitness)

            if verbose and generation % 5 == 0:
                print(f"Gen {generation:2d}: Best={generation_best_fitness:.1f}, Thresholds={generation_best_individual}")

            selected = self.selection(population, fitness_scores)

            next_generation = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % len(selected)]

                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)

                next_generation.extend([offspring1, offspring2])

            population = next_generation[:self.population_size]

        if verbose:
            print(f"Optimal thresholds: {self.best_solution}")
        

        return self.best_solution if self.best_solution is not None else (self.min_threshold, self.max_threshold)


class EnhancedLaneDepartureWarning:
    """
    Enhanced Lane Departure Warning System with Kalman Filtering and Genetic Algorithm Optimization

    Processing Pipeline:
    1. Image preprocessing with color enhancement
    2. Shadow detection and mitigation
    3. Adaptive Canny edge detection (optimized every 120th frame)
    4. Adaptive ROI based on vanishing point
    5. Hough transform line detection
    6. Line validation and filtering
    7. Kalman filter tracking
    8. Lane departure analysis
    """

    def __init__(self):
        # Image processing parameters (will be optimized by genetic algorithm)
        self.gaussian_kernel = 5
        self.canny_low = 50
        self.canny_high = 150
        self.shadow_threshold = 0.1

        # Hough transform parameters
        self.rho = 1
        self.theta = np.pi / 180
        self.threshold = 20
        self.min_line_length = 80
        self.max_line_gap = 50

        # Lane classification parameters
        self.min_lane_slope = 0.3
        self.max_lane_slope = 3.0

        # Kalman filter parameters
        self.kalman_dt = 1.0
        self.process_noise = 0.01
        self.measurement_noise = 0.1

        # Initialize Kalman filters for tracking
        self.left_lane_kalman = self.initialize_kalman_filter()
        self.right_lane_kalman = self.initialize_kalman_filter()

        # Genetic Algorithm Optimizer
        self.genetic_optimizer = GeneticThresholdOptimizer(
            population_size=20,  # Reduced for faster optimization
            generations=10,      # Reduced for real-time performance
            mutation_rate=0.1
        )

        # Frame counting for genetic algorithm optimization
        self.frame_count = 0
        self.optimization_interval = 120  # Optimize every 120 frames
        self.last_optimization_frame = 0

    def initialize_kalman_filter(self):
        """Initialize Kalman filter with state [m, c, dm, dc] for slope and intercept tracking"""
        kalman = cv2.KalmanFilter(4, 2)

        # State transition matrix F (constant velocity model)
        kalman.transitionMatrix = np.array([[1, 0, self.kalman_dt, 0],
                                          [0, 1, 0, self.kalman_dt],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]], dtype=np.float32)

        # Measurement matrix H
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0]], dtype=np.float32)

        # Process noise covariance Q
        kalman.processNoiseCov = self.process_noise * np.eye(4, dtype=np.float32)

        # Measurement noise covariance R
        kalman.measurementNoiseCov = self.measurement_noise * np.eye(2, dtype=np.float32)

        # Initial state estimate
        kalman.statePre = np.array([0, 0, 0, 0], dtype=np.float32)
        kalman.statePost = np.array([0, 0, 0, 0], dtype=np.float32)

        # Initial error covariance
        kalman.errorCovPre = np.eye(4, dtype=np.float32)
        kalman.errorCovPost = np.eye(4, dtype=np.float32)

        return kalman

    def optimize_canny_thresholds(self, image):
        """Optimize Canny thresholds using genetic algorithm"""
        if image is None or image.size == 0:
            print("⚠️  Cannot optimize: Invalid input image")
            return

        print(f"Frame {self.frame_count}: Running genetic algorithm optimization...")
        # start_time = time.time()

        try:
            optimal_thresholds = self.genetic_optimizer.evolve(image, verbose=False)

            if optimal_thresholds:
                self.canny_low, self.canny_high = optimal_thresholds
                # optimization_time = time.time() - start_time
                # print(f" Optimization complete")
                print(f"   New thresholds: Low={self.canny_low}, High={self.canny_high}")
            else:
                print("⚠️  Optimization failed, keeping current thresholds")
        except Exception as e:
            print(f"❌ Optimization error: {e}")
            print("   Keeping current thresholds")

    def preprocess_image(self, image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Enhanced preprocessing with color-based lane enhancement
        Args:
            image (np.ndarray): Input image to preprocess
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Preprocessed images (gray, enhanced_gray, blur)
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is None or empty")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Enhance white and yellow lane markings
        enhanced_gray = self.enhance_lane_markings(image, gray, hsv)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(enhanced_gray, (self.gaussian_kernel, self.gaussian_kernel), 0)

        return gray, enhanced_gray, blur

    def enhance_lane_markings(self, bgr_image, gray, hsv) -> np.ndarray:
        """Color-based lane marking enhancement for white and yellow lanes"""
        # Create masks for white and yellow lane markings
        white_mask = self.create_white_lane_mask(bgr_image, hsv)
        yellow_mask = self.create_yellow_lane_mask(hsv)

        # Combine masks
        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # Clean up the mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)

        # Enhance lane markings in grayscale image
        enhanced = gray.copy()
        enhanced[lane_mask > 0] = np.minimum(enhanced[lane_mask > 0] + 50, 255)

        return enhanced

    def create_white_lane_mask(self, bgr_image, hsv) -> np.ndarray:
        """Create mask for white lane markings"""
        # White in BGR
        lower_white_bgr = np.array([180, 180, 180])
        upper_white_bgr = np.array([255, 255, 255])
        white_mask_bgr = cv2.inRange(bgr_image, lower_white_bgr, upper_white_bgr)

        # White in HSV
        lower_white_hsv = np.array([0, 0, 200])
        upper_white_hsv = np.array([180, 30, 255])
        white_mask_hsv = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)

        return cv2.bitwise_or(white_mask_bgr, white_mask_hsv)

    def create_yellow_lane_mask(self, hsv) -> np.ndarray:
        """Create mask for yellow lane markings"""
        lower_yellow = np.array([15, 80, 100])
        upper_yellow = np.array([35, 255, 255])
        return cv2.inRange(hsv, lower_yellow, upper_yellow)

    def shadow_mitigation_hsv(self, image: np.ndarray) -> tuple:
        """
        Shadow detection and mitigation using HSV color space

        Implements:
        - NSVDI(i, j) = (S(i, j) - V(i, j)) / (S(i, j) + V(i, j))
        - Linear Correlation Correction (LCC)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0].astype(np.float32), hsv[:, :, 1].astype(np.float32), hsv[:, :, 2].astype(np.float32)

        # Normalized Saturation-Value Difference Index
        denominator = S + V
        denominator[denominator == 0] = 1e-10
        nsvdi = (S - V) / denominator

        # Binary shadow mask segmentation
        shadow_mask = nsvdi > self.shadow_threshold

        # Linear Correlation Correction for shadow regions
        corrected_image = image.copy().astype(np.float32)

        if np.any(shadow_mask):
            shadow_pixels = V[shadow_mask]
            non_shadow_pixels = V[~shadow_mask]

            if len(non_shadow_pixels) > 0 and len(shadow_pixels) > 0:
                mu_s = np.mean(shadow_pixels)
                sigma_s = np.std(shadow_pixels)
                mu_ns = np.mean(non_shadow_pixels)
                sigma_ns = np.std(non_shadow_pixels)

                if sigma_s > 0:
                    correction_factor = sigma_ns / sigma_s
                    V_corrected = V.copy()
                    V_corrected[shadow_mask] = (mu_ns - mu_s) + correction_factor * (V[shadow_mask] - mu_s)

                    hsv_corrected = hsv.copy()
                    hsv_corrected[:, :, 2] = np.clip(V_corrected, 0, 255).astype(np.uint8)
                    corrected_image = cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)

        return corrected_image.astype(np.uint8), shadow_mask.astype(np.uint8)

    def canny_edge_detection(self, blurred_image, low_thresh=None, high_thresh=None) -> np.ndarray:
        """Apply Canny edge detection with optimized thresholds"""
        if low_thresh is None:
            low_thresh = self.canny_low
        if high_thresh is None:
            high_thresh = self.canny_high

        return cv2.Canny(blurred_image, low_thresh, high_thresh)

    def adaptive_roi_vanishing_point(self, image) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray]:
        """Vectorized adaptive ROI based on vanishing point detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)  
        
        if lines is None or len(lines) < 2:
            height, width = image.shape[:2]
            vanishing_point = (width // 2, height // 2)
        else:
            lines = lines.reshape(-1, 2)  # Reshape from (n,1,2) to (n,2)
            vanishing_point = self.estimate_vanishing_point(lines, image.shape)
        
        # Vectorized ROI creation
        roi_mask = self.create_adaptive_roi(image.shape, vanishing_point)
        
        return roi_mask, vanishing_point, edges

    def estimate_vanishing_point(self, lines: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Vectorized vanishing point estimation using homogeneous coordinates"""
        height, width = image_shape[:2]
        
        if len(lines) < 2:
            return (width // 2, height // 3)
        
        # Limit lines for performance
        max_lines = min(100, len(lines))  
        lines = lines[:max_lines]
        
        rho = lines[:, 0]
        theta = lines[:, 1]
        
        # Vectorized homogeneous line representation
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Create all pairwise combinations efficiently
        n = len(lines)
        i_indices, j_indices = np.triu_indices(n, k=1)  # Upper triangle indices
        
        # Vectorized line parameters for all pairs
        cos1, cos2 = cos_theta[i_indices], cos_theta[j_indices]
        sin1, sin2 = sin_theta[i_indices], sin_theta[j_indices]
        rho1, rho2 = rho[i_indices], rho[j_indices]
        
        # Vectorized cross product for line intersections
        # l1 = [cos1, sin1, -rho1], l2 = [cos2, sin2, -rho2]
        # v = l1 × l2
        v_x = sin1 * (-rho2) - (-rho1) * sin2  # sin1*(-rho2) - (-rho1)*sin2
        v_y = (-rho1) * cos2 - cos1 * (-rho2)  # (-rho1)*cos2 - cos1*(-rho2) 
        v_z = cos1 * sin2 - sin1 * cos2
        
        # Filter valid intersections (non-parallel lines)
        valid_mask = np.abs(v_z) > 1e-6
        
        if not np.any(valid_mask):
            return (width // 2, height // 3)
        
        # Convert to Cartesian coordinates
        x_intersect = v_x[valid_mask] / v_z[valid_mask]
        y_intersect = v_y[valid_mask] / v_z[valid_mask]
        
        # Filter points within reasonable bounds
        bounds_mask = ((x_intersect >= 0) & (x_intersect <= width) & 
                    (y_intersect >= 0) & (y_intersect <= height))
        
        if not np.any(bounds_mask):
            return (width // 2, height // 3)
        
        valid_x = x_intersect[bounds_mask]
        valid_y = y_intersect[bounds_mask]
        
        # Use median instead of DBSCAN for speed
        vp_x = int(np.median(valid_x))
        vp_y = int(np.median(valid_y))
        
        return (vp_x, vp_y)

        # Use DBSCAN clustering to find main cluster
        if len(intersections) > 1:
            intersections_array = np.array(intersections)
            clustering = DBSCAN(eps=10, min_samples=2).fit(intersections_array)

            labels = clustering.labels_
            if len(set(labels)) > 1 and -1 in labels:
                unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
                if len(unique_labels) > 0:
                    main_cluster_label = unique_labels[np.argmax(counts)]
                    cluster_points = intersections_array[labels == main_cluster_label]
                    vp = np.mean(cluster_points, axis=0).astype(int)
                    return tuple(vp)

        # Fallback: use median of all intersections
        intersections_array = np.array(intersections)
        vp = np.median(intersections_array, axis=0).astype(int)
        return tuple(vp)

    def create_adaptive_roi(self, image_shape, vanishing_point) -> np.ndarray:
        """Vectorized adaptive ROI creation using mesh operations"""
        height, width = image_shape[:2]
        vx, vy = vanishing_point
        
        # Ensure reasonable vanishing point
        vy = max(50, min(vy, height - 100))
        
        # ROI parameters
        h_offset = 50
        w_top = width // 8
        w_bottom = width // 2
        
        # Trapezoid vertices
        vertices = np.array([
            [max(0, vx - w_top), max(0, vy + h_offset)],      # top_left
            [min(width, vx + w_top), max(0, vy + h_offset)],  # top_right  
            [min(width, vx + w_bottom), height],              # bottom_right
            [max(0, vx - w_bottom), height]                   # bottom_left
        ], dtype=np.int32)
        
        # Vectorized mask creation
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [vertices], (255,))
        
        return roi_mask

    def region_of_interest(self, image) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray]:
        """Apply adaptive ROI based on vanishing point detection"""
        #start = time.time()
        roi_mask, vanishing_point, edges_for_vp = self.adaptive_roi_vanishing_point(image)

        # Apply ROI mask
        masked_image = cv2.bitwise_and(image, image, mask=roi_mask)

        #end = time.time()
        # Only print timing for optimization frames
        #if self.frame_count % self.optimization_interval == 0:
        #    print(f"Adaptive ROI computation time: {end - start:.4f} seconds")

        return masked_image, vanishing_point, roi_mask

    def hough_transform(self, edges) -> np.ndarray:
        """Apply Hough Transform for line detection"""
        return cv2.HoughLinesP(
            edges,
            rho=self.rho,
            theta=self.theta,
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

    def line_validation(self, lines: np.ndarray, vanishing_point: Tuple[int, int],
                              image_shape: Tuple[int, int], image: np.ndarray) -> np.ndarray:
        """
        Line validation for lane detection
        """
        if len(lines) == 0:
            return np.empty((0, 4), dtype=np.int32)

        height, width = image_shape[:2]
        vx, vy = vanishing_point
        
        # Extract line coordinates 
        #lines_reshaped = lines.reshape(-1, 4)  # Ensure shape is (N, 4)
        #x1, y1, x2, y2 = lines_reshaped.T  # Transpose so we get individual coordinate arrays
        x1, y1, x2, y2 = lines.T

        # Calculate line lengths
        line_lengths = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Filter out short lines
        length_mask = line_lengths >= 50
        
        # Calculate slopes 
        dx = x2 - x1
        dy = y2 - y1
        
        # Avoid division by zero
        valid_dx_mask = np.abs(dx) > 10
        
        slopes = np.full(len(lines), np.inf)
        valid_slope_indices = valid_dx_mask
        slopes[valid_slope_indices] = dy[valid_slope_indices] / dx[valid_slope_indices]
        
        # Filter slopes within range
        slope_mask = (-3 < slopes) & (slopes < 3)
        
        # Calculate distance to vanishing point
        a = dy
        b = -dx
        c = dx * y1 - dy * x1
        
        # Avoid division by zero in distance calculation
        denominator = np.sqrt(a**2 + b**2)
        valid_denom_mask = denominator > 0
        
        distances_to_vp = np.full(len(lines), np.inf)
        distances_to_vp[valid_denom_mask] = (
            np.abs(a[valid_denom_mask] * vx + b[valid_denom_mask] * vy + c[valid_denom_mask]) / 
            denominator[valid_denom_mask]
        )
        
        # Filter by distance to vanishing point
        vp_distance_mask = distances_to_vp <= 100
        
        # Combine all masks
        overall_mask = length_mask & slope_mask & vp_distance_mask & valid_dx_mask & valid_denom_mask
        
        if not np.any(overall_mask):
            return np.empty((0, 4), dtype=np.int32)
        
        # Filter lines based on combined mask
        #valid_lines_coords = lines_reshaped[overall_mask]
        valid_lines_coords = lines[overall_mask]
        valid_lengths = line_lengths[overall_mask]
        valid_distances = distances_to_vp[overall_mask]
        
        # Vectorized color scoring
        color_scores = self._calculate_color_scores_vectorized(valid_lines_coords, image, width, height)
        
        # Calculate confidence scores 
        w1, w2, w3, w4 = 0.3, 0.3, 0.2, 0.2
        
        accumulator_scores = np.ones(len(valid_lines_coords))  
        geometric_scores = 1.0 / (1.0 + valid_distances / 50.0)
        length_scores = valid_lengths / 100.0
        normalized_color_scores = color_scores / 255.0
        
        confidence_scores = (w1 * accumulator_scores +
                            w2 * length_scores +
                            w3 * geometric_scores +
                            w4 * normalized_color_scores)
        
        if len(confidence_scores) > 10:
            top_indices = np.argsort(confidence_scores)[-10:]  # Top 10
            return valid_lines_coords[top_indices]

        return valid_lines_coords

    def _calculate_color_scores_vectorized(self, lines_coords: np.ndarray, image: np.ndarray, 
                                        width: int, height: int) -> np.ndarray:
        """
        Vectorized color scoring for multiple lines
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color_scores = np.zeros(len(lines_coords))
        
        for i, (x1, y1, x2, y2) in enumerate(lines_coords):
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            num_samples = max(int(line_length // 10), 10)
            
            # Generate sample points along the line
            t = np.linspace(0, 1, num_samples)
            x_samples = (x1 + t * (x2 - x1)).astype(np.int32)
            y_samples = (y1 + t * (y2 - y1)).astype(np.int32)   
            
            # Filter valid samples
            valid_mask = ((x_samples >= 0) & (x_samples < width) & 
                        (y_samples >= 0) & (y_samples < height))
            
            if np.any(valid_mask):
                valid_x = x_samples[valid_mask]
                valid_y = y_samples[valid_mask]
                color_scores[i] = np.mean(gray[valid_y, valid_x])
        
        return color_scores

    def kalman_tracking(self, detected_lines: List[Tuple[float, float, float]]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """
        Temporal lane tracking with Kalman filtering

        Implements:
        - State vector: x_k = [m, c, ṁ, ċ]ᵀ
        - Prediction: x̂_k|k-1 = F * x̂_k-1|k-1
        - Update: Kalman gain and correction equations
        """
        # Predict first for both filters
        left_predicted = self.left_lane_kalman.predict()
        right_predicted = self.right_lane_kalman.predict()
        
        if not detected_lines:
            # No measurements available - use predicted values
            left_lane = (float(left_predicted[0]), float(left_predicted[1]))
            right_lane = (float(right_predicted[0]), float(right_predicted[1]))
            return left_lane, right_lane

        # Separate lines into left and right candidates
        left_candidates = []
        right_candidates = []

        for rho, theta, confidence in detected_lines:
            if abs(np.sin(theta)) > 0.1:
                slope = -np.cos(theta) / np.sin(theta)
                intercept = rho / np.sin(theta)

                if slope < 0:
                    left_candidates.append((slope, intercept, confidence))
                else:
                    right_candidates.append((slope, intercept, confidence))

        # Process left lane
        if np.any(left_candidates):
            # Have measurement - use correction step
            left_candidates.sort(key=lambda x: x[2], reverse=True)
            slope, intercept = left_candidates[0][:2]
            
            measurement = np.array([[slope], [intercept]], dtype=np.float32)
            corrected = self.left_lane_kalman.correct(measurement)
            left_lane = (float(corrected[0]), float(corrected[1]))
        else:
            # No measurement - use prediction only
            left_lane = (float(left_predicted[0]), float(left_predicted[1]))

        # Process right lane
        if np.any(right_candidates):
            # Have measurement - use correction step
            right_candidates.sort(key=lambda x: x[2], reverse=True)
            slope, intercept = right_candidates[0][:2]
            
            measurement = np.array([[slope], [intercept]], dtype=np.float32)
            corrected = self.right_lane_kalman.correct(measurement)
            right_lane = (float(corrected[0]), float(corrected[1]))
        else:
            # No measurement - use prediction only
            right_lane = (float(right_predicted[0]), float(right_predicted[1]))

        return left_lane, right_lane

    def classify_lanes(self, lines : np.ndarray, 
                       image_shape : Tuple[int, int], 
                       vanishing_point : Tuple[int, int], 
                       image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Classify detected lines into left and right lanes.
        Args:
            lines: Detected lines from Hough transform
            image_shape: Shape of the input image
            vanishing_point: Estimated vanishing point
            image: Input image for validation
        
        Returns:
            Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]
            Left and right lane lines
        """
        if lines is None or len(lines) == 0:
            return np.empty((0, 4), dtype=np.int32), np.empty((0, 4), dtype=np.int32)


        height, width = image_shape[:2]
        image_center_x = width // 2
        min_distance_from_center: float = width * 0.1

        # Reshape lines to ensure consistent format (N, 4)
        if lines.ndim == 3:
            lines_reshaped = lines.reshape(-1, 4)  # From (N, 1, 4) to (N, 4)
        else:
            lines_reshaped = lines.copy()

        # Extract coordinates
        x1, y1, x2, y2 = lines_reshaped.T

        # Calculate dx and filter out vertical lines
        dx = x2 - x1
        valid_dx_mask = np.abs(dx) >= 10

        if not np.any(valid_dx_mask):  
            return np.empty((0, 4), dtype=np.int32), np.empty((0, 4), dtype=np.int32)

        # Filter lines with sufficient dx
        valid_lines = lines_reshaped[valid_dx_mask]
        x1_valid, y1_valid, x2_valid, y2_valid = valid_lines.T
        dx_valid = x2_valid - x1_valid
        dy_valid = y2_valid - y1_valid

        # Calculate slopes - vectorized (safe since we filtered dx >= 10)
        slopes = dy_valid / dx_valid

        # Filter by slope range - vectorized
        slope_mask = (np.abs(slopes) >= self.min_lane_slope) & (np.abs(slopes) <= self.max_lane_slope)

        if not np.any(slope_mask):  
            return np.empty((0, 4), dtype=np.int32), np.empty((0, 4), dtype=np.int32)

        # Apply slope filtering
        slope_filtered_lines = valid_lines[slope_mask]
        slopes_filtered = slopes[slope_mask]

        if len(slope_filtered_lines) == 0:
            return np.empty((0, 4), dtype=np.int32), np.empty((0, 4), dtype=np.int32)

        # Calculate line centers
        x1_s, y1_s, x2_s, y2_s = slope_filtered_lines.T
        line_centers_x = (x1_s + x2_s) / 2

        # Position and distance filters
        center_distances = np.abs(line_centers_x - image_center_x)
        far_from_center_mask = center_distances >= min_distance_from_center

        # Left/right classification
        left_mask = (slopes_filtered < 0) & (line_centers_x < image_center_x * 1.2) & far_from_center_mask
        right_mask = (slopes_filtered > 0) & (line_centers_x > image_center_x * 0.8) & far_from_center_mask

        # Extract classified lines
        left_lines_raw = slope_filtered_lines[left_mask]
        right_lines_raw = slope_filtered_lines[right_mask]

        # Apply validation (keeping numpy format)
        left_lines_validated = self.line_validation(left_lines_raw, vanishing_point, image_shape, image)
        right_lines_validated = self.line_validation(right_lines_raw, vanishing_point, image_shape, image)

        return left_lines_validated, right_lines_validated

    def lane_departure_warning(self, left_lines: np.ndarray,
                               right_lines: np.ndarray,
                               image_shape: Tuple[int, int]) -> str:
        """Analyze lane positions for departure warning"""
        height, width = image_shape[:2]

        if not np.any(left_lines) and not np.any(right_lines):
            return "CRITICAL: NO LANES DETECTED"
        elif not np.any(left_lines):
            return "WARNING: LEFT LANE LOST - POTENTIAL DEPARTURE"
        elif not np.any(right_lines):
            return "WARNING: RIGHT LANE LOST - POTENTIAL DEPARTURE"
        else:
            return self.analyze_lane_departure(left_lines, right_lines, image_shape)

    def analyze_lane_departure(self, left_lines: np.ndarray, right_lines: np.ndarray, 
                               image_shape: Tuple[int, int]) -> str:
        """Vectorized lane departure analysis based on vehicle position"""
        height, width = image_shape[:2]
        sample_y = height - height // 3

        # Convert line lists to numpy arrays for vectorized operations
        def calculate_positions_vectorized(lines_array: np.ndarray) -> np.ndarray:
            if len(lines_array) == 0:
                return np.array([])
            
                        
            x1, y1, x2, y2 = lines_array.T  # Shape: (4, n_lines) -> 4 arrays of length n_lines
            
            # Vectorized check for non-horizontal lines
            valid_mask = y1 != y2
            
            if not np.any(valid_mask):
                return np.array([])
            
            # Filter valid lines
            x1_valid = x1[valid_mask]
            y1_valid = y1[valid_mask]
            x2_valid = x2[valid_mask]
            y2_valid = y2[valid_mask]
            
            # Vectorized intersection calculation
            # x_at_sample = x1 + (x2 - x1) * (sample_y - y1) / (y2 - y1)
            t = (sample_y - y1_valid) / (y2_valid - y1_valid)
            x_at_sample = x1_valid + (x2_valid - x1_valid) * t
            
            # Filter positions within image bounds
            bounds_mask = (x_at_sample >= 0) & (x_at_sample <= width)
            
            return x_at_sample[bounds_mask]

        # Calculate positions for both left and right lanes
        left_x_positions = calculate_positions_vectorized(left_lines)
        right_x_positions = calculate_positions_vectorized(right_lines)

        # Early returns for missing data
        if len(left_x_positions) == 0 and len(right_x_positions) == 0:
            return "UNKNOWN: LANE POSITIONS UNCLEAR"

        # Calculate departure ratio if both lanes are present
        if len(left_x_positions) > 0 and len(right_x_positions) > 0:
            # Mean calculations
            avg_left_x = np.mean(left_x_positions)
            avg_right_x = np.mean(right_x_positions)
            
            lane_center = (avg_left_x + avg_right_x) / 2
            vehicle_center = width / 2
            
            offset = vehicle_center - lane_center
            lane_width = avg_right_x - avg_left_x

            if lane_width > 0:
                departure_ratio = abs(offset) / (lane_width / 2)

                # Threshold checks
                if departure_ratio > 0.7:
                    direction = "LEFT" if offset < 0 else "RIGHT"
                    return f"DANGER: {direction} DEPARTURE DETECTED ({departure_ratio:.1%})"
                elif departure_ratio > 0.4:
                    direction = "LEFT" if offset < 0 else "RIGHT"
                    return f"CAUTION: DRIFTING {direction} ({departure_ratio:.1%})"
                else:
                    return "SAFE: CENTERED IN LANE"

        return "SAFE: LANES DETECTED"

    def reset_kalman_filters(self) -> None:
        """Reset Kalman filters when tracking is lost"""
        print("WARNING: KALMAN FILTERS RESET")
        self.left_lane_kalman = self.initialize_kalman_filter()
        self.right_lane_kalman = self.initialize_kalman_filter()

    def extract_slope_intercept_vectorized(self, lines) -> List[Tuple[float, float, float]]:
        """Vectorized extraction of slope and intercept from lines"""
        if lines is None or len(lines) == 0:
            return []
        
        # Ensure proper shape
        if lines.ndim == 3:
            lines = lines.reshape(-1, 4)
        
        x1, y1, x2, y2 = lines.T
        
        # Vectorized slope calculation - filter valid lines
        dx = x2 - x1
        valid_mask = np.abs(dx) > 1e-6
        
        if not np.any(valid_mask):
            return []
        
        # Extract valid coordinates
        x1_valid = x1[valid_mask]
        y1_valid = y1[valid_mask]
        x2_valid = x2[valid_mask] 
        y2_valid = y2[valid_mask]
        dx_valid = x2_valid - x1_valid
        dy_valid = y2_valid - y1_valid
        
        # Vectorized slope and intercept calculation
        slopes = dy_valid / dx_valid
        intercepts = y1_valid - slopes * x1_valid
        confidences = np.ones(len(slopes))  # All lines get confidence 1.0
        
        # Return as list of tuples for compatibility with Kalman tracking
        return list(zip(slopes, intercepts, confidences))

    def convert_lines_to_polar(self, lines):
      """Convert Cartesian lines to polar format for Kalman tracking"""
      polar_lines = []

      for line in lines:
          x1, y1, x2, y2 = line

          # Calculate line parameters
          if abs(x2 - x1) > 1e-6:  # Avoid division by zero
              # Convert to polar form: rho = x*cos(theta) + y*sin(theta)
              dx = x2 - x1
              dy = y2 - y1

              # Calculate angle
              theta = np.arctan2(dy, dx)

              # Calculate rho (distance from origin to line)
              rho = x1 * np.cos(theta) + y1 * np.sin(theta)

              # Assign confidence score
              confidence = 1.0

              polar_lines.append((rho, theta, confidence))

      return polar_lines

    def draw_lanes(self, image, left_lines, right_lines) -> np.ndarray:
        """Draw detected lanes on image"""
        line_image = np.zeros_like(image)

        # Draw left lanes in red
        for i, line in enumerate(left_lines):
            x1, y1, x2, y2 = line
            thickness = 8 if i == 0 else 4
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), thickness)

        # Draw right lanes in green
        for i, line in enumerate(right_lines):
            x1, y1, x2, y2 = line
            thickness = 8 if i == 0 else 4
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), thickness)

        return cv2.addWeighted(image, 0.8, line_image, 1.0, 0)

    def process_image(self, image: np.ndarray):
        """Main processing pipeline with genetic algorithm optimization"""
        if image is None or image.size == 0:
            print("❌ Error: Invalid input image")
            return None

        self.frame_count += 1

        try:
            # Run genetic algorithm optimization every 120th frame
            if (self.frame_count - self.last_optimization_frame) >= self.optimization_interval:
                self.optimize_canny_thresholds(image)
                self.last_optimization_frame = self.frame_count

            # Stage 1: Preprocessing with color enhancement
            gray, enhanced_gray, blur = self.preprocess_image(image)

            # Stage 2: Shadow mitigation (optional)
            # corrected_image, shadow_mask = self.shadow_mitigation_hsv(image)

            # Stage 3: Edge detection with optimized thresholds
            edges = self.canny_edge_detection(blur)

            # Stage 4: Adaptive ROI
            roi_edges, vanishing_point, roi_mask = self.region_of_interest(edges)

            # Stage 5: Line detection
            lines = self.hough_transform(roi_edges)

            # Stage 6: Lane classification and validation
            left_lines, right_lines = self.classify_lanes(lines, image.shape, vanishing_point, image)

            # Stage 7: Kalman tracking
            if len(left_lines) > 0 or len(right_lines) > 0:
    
                all_lines = np.vstack([left_lines, right_lines]) if len(left_lines) > 0 and len(right_lines) > 0 else \
                            left_lines if len(left_lines) > 0 else right_lines
                
                detected_lines_slope_intercept = self.extract_slope_intercept_vectorized(all_lines)
                tracked_left, tracked_right = self.kalman_tracking(detected_lines_slope_intercept)

            # Stage 8: Departure analysis
            warning = self.lane_departure_warning(left_lines, right_lines, image.shape)

            # Visualization
            result_image = self.draw_lanes(image, left_lines, right_lines)

            # Add status text with genetic algorithm info
            cv2.putText(result_image, warning, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_image, f"Left: {len(left_lines)}, Right: {len(right_lines)}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(result_image, f"VP: ({vanishing_point[0]},{vanishing_point[1]})",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(result_image, f"Thresholds: {self.canny_low},{self.canny_high}",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            cv2.putText(result_image, f"Frame: {self.frame_count}",
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

            # Show genetic algorithm status
            frames_until_next_opt = self.optimization_interval - (self.frame_count - self.last_optimization_frame)
            cv2.putText(result_image, f"Next GA opt in: {frames_until_next_opt} frames",
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 255), 1)

            return {
                'result': result_image,
                'original': image,
                'gray': gray,
                'enhanced': enhanced_gray,
                'edges': edges,
                'roi_edges': roi_edges,
                'roi_mask': roi_mask,
                'vanishing_point': vanishing_point,
                'left_lines': left_lines,
                'right_lines': right_lines,
                'warning': warning,
                'canny_thresholds': (self.canny_low, self.canny_high),
                'frame_count': self.frame_count
            }

        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return None

    def process_video(self, video_path, output_path):
        """Process video with lane detection and genetic algorithm optimization"""
        if not video_path or not output_path:
            print("❌ Error: Invalid video paths provided")
            return

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Error: Could not open video file: {video_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or width <= 0 or height <= 0:
            print("❌ Error: Invalid video properties")
            cap.release()
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"🎬 Processing video: {total_frames} frames")
        print(f"🧬 Genetic optimization will run every {self.optimization_interval} frames")

        optimization_count = 0
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_image(frame)

            if result is not None:
                out.write(result['result'])

                frame_count += 1

                # Count optimizations
                if (frame_count - self.last_optimization_frame) == 0 and frame_count > 1:
                    optimization_count += 1

                # Progress reporting
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"📊 Progress: {progress:.1f}% ({frame_count}/{total_frames}) - Optimizations: {optimization_count}")

                # Show current thresholds during optimization frames
                if frame_count % self.optimization_interval == 0:
                    print(f"🎯 Current Canny thresholds: Low={self.canny_low}, High={self.canny_high}")
            else:
                print(f"⚠️  Skipping frame {frame_count} due to processing error")

        cap.release()
        out.release()

        print(f"✅ Enhanced lane detection video saved: {output_path}")
        print(f"🧬 Total genetic algorithm optimizations: {optimization_count}")
        print(f"📈 Final optimized thresholds: Low={self.canny_low}, High={self.canny_high}")

    def process_realtime_camera(self, camera_index=0):
        """Process real-time camera feed with genetic algorithm optimization"""
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return

        print("🎥 Starting real-time lane detection with genetic algorithm optimization")
        print("📋 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 'o' to force optimization")
        print("   - Press 'r' to reset thresholds to default")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break

            # Process frame
            result = self.process_image(frame)

            # Display result
            cv2.imshow('Enhanced Lane Detection with Genetic Algorithm', result['result'])

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('o'):
                print("🧬 Manual optimization triggered...")
                self.optimize_canny_thresholds(frame)
            elif key == ord('r'):
                print("🔄 Resetting thresholds to default...")
                self.canny_low = 50
                self.canny_high = 150
                print(f"   Reset to: Low={self.canny_low}, High={self.canny_high}")

        cap.release()
        cv2.destroyAllWindows()
        print(f"🏁 Session ended. Final thresholds: Low={self.canny_low}, High={self.canny_high}")


