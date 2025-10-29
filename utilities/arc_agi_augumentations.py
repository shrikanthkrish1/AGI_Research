"""
Structural Augmentations for ARC-AGI (MITS Strategy)
=====================================================
Creates 32+ structural transformations of input/output grids.
Provides `prepare_augmentations_for_mits` so tests/imports work.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Callable
from itertools import permutations
import copy
import ast

class StructuralAugmenter:
    """
    Generates structural augmentations for ARC-AGI tasks.
    """

    def __init__(self, num_augmentations: int = 32, seed: Optional[int] = None):
        self.num_augmentations = num_augmentations
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ---------------- Core transforms ----------------
    @staticmethod
    def rotate_90(grid: List[List[int]]) -> List[List[int]]:
        return np.rot90(np.array(grid), k=-1).tolist()

    @staticmethod
    def rotate_180(grid: List[List[int]]) -> List[List[int]]:
        return np.rot90(np.array(grid), k=-2).tolist()

    @staticmethod
    def rotate_270(grid: List[List[int]]) -> List[List[int]]:
        return np.rot90(np.array(grid), k=-3).tolist()

    @staticmethod
    def flip_horizontal(grid: List[List[int]]) -> List[List[int]]:
        return np.fliplr(np.array(grid)).tolist()

    @staticmethod
    def flip_vertical(grid: List[List[int]]) -> List[List[int]]:
        return np.flipud(np.array(grid)).tolist()

    @staticmethod
    def transpose(grid: List[List[int]]) -> List[List[int]]:
        return np.array(grid).T.tolist()

    @staticmethod
    def transpose_anti(grid: List[List[int]]) -> List[List[int]]:
        arr = np.array(grid)
        return np.flip(arr.T, axis=(0, 1)).tolist()

    @staticmethod
    def identity(grid: List[List[int]]) -> List[List[int]]:
        return copy.deepcopy(grid)

    @staticmethod
    def permute_colors(grid: List[List[int]], color_map: Dict[int, int]) -> List[List[int]]:
        arr = np.array(grid)
        result = np.zeros_like(arr)
        for old_color, new_color in color_map.items():
            result[arr == old_color] = new_color
        return result.tolist()

    # ---------------- D8 group ----------------
    @staticmethod
    def get_d8_transforms() -> List[Tuple[str, callable]]:
        return [
            ('identity', StructuralAugmenter.identity),
            ('rotate_90', StructuralAugmenter.rotate_90),
            ('rotate_180', StructuralAugmenter.rotate_180),
            ('rotate_270', StructuralAugmenter.rotate_270),
            ('flip_h', StructuralAugmenter.flip_horizontal),
            ('flip_v', StructuralAugmenter.flip_vertical),
            ('transpose', StructuralAugmenter.transpose),
            ('transpose_anti', StructuralAugmenter.transpose_anti)
        ]

    @staticmethod
    def get_inverse_transform(transform_name: str) -> callable:
        inverse_map = {
            'identity': StructuralAugmenter.identity,
            'rotate_90': StructuralAugmenter.rotate_270,
            'rotate_180': StructuralAugmenter.rotate_180,
            'rotate_270': StructuralAugmenter.rotate_90,
            'flip_h': StructuralAugmenter.flip_horizontal,
            'flip_v': StructuralAugmenter.flip_vertical,
            'transpose': StructuralAugmenter.transpose,
            'transpose_anti': StructuralAugmenter.transpose_anti
        }
        return inverse_map.get(transform_name, StructuralAugmenter.identity)

    # ---------------- color permutations ----------------
    @staticmethod
    def generate_random_color_permutation(keep_background: bool = True) -> Dict[int, int]:
        colors = list(range(10))
        if keep_background:
            other_colors = colors[1:]
            random.shuffle(other_colors)
            permuted = [0] + other_colors
        else:
            permuted = colors.copy()
            random.shuffle(permuted)
        return {i: permuted[i] for i in range(10)}

    # ---------------- task transformation ----------------
    def transform_task(
        self,
        task: Dict,
        geometric_transform: callable,
        color_map: Optional[Dict[int, int]] = None,
        shuffle_examples: bool = False
    ) -> Dict:
        transformed_task = {'train': [], 'test': []}
        train_examples = task.get('train', [])
        if shuffle_examples:
            train_examples = train_examples.copy()
            random.shuffle(train_examples)

        for example in train_examples:
            transformed_input = geometric_transform(example['input'])
            transformed_output = geometric_transform(example['output'])
            if color_map:
                transformed_input = self.permute_colors(transformed_input, color_map)
                transformed_output = self.permute_colors(transformed_output, color_map)
            transformed_task['train'].append({
                'input': transformed_input,
                'output': transformed_output
            })

        for example in task.get('test', []):
            transformed_input = geometric_transform(example['input'])
            if color_map:
                transformed_input = self.permute_colors(transformed_input, color_map)
            test_example = {'input': transformed_input}
            if 'output' in example:
                transformed_output = geometric_transform(example['output'])
                if color_map:
                    transformed_output = self.permute_colors(transformed_output, color_map)
                test_example['output'] = transformed_output
            transformed_task['test'].append(test_example)

        return transformed_task

    # ---------------- generate augmentations ----------------
    def generate_augmentations(
        self,
        task: Dict,
        include_identity: bool = True
    ) -> List[Dict]:
        augmentations = []
        d8_transforms = self.get_d8_transforms()
        color_permutations = [
            None,
            self.generate_random_color_permutation(keep_background=True),
            self.generate_random_color_permutation(keep_background=True),
            self.generate_random_color_permutation(keep_background=False),
        ]

        aug_id = 0
        for transform_name, transform_func in d8_transforms:
            for color_idx, color_map in enumerate(color_permutations):
                if not include_identity and transform_name == 'identity' and color_map is None:
                    continue

                transformed_task = self.transform_task(
                    task,
                    transform_func,
                    color_map,
                    shuffle_examples=(aug_id % 2 == 1)
                )

                inverse_geometric = self.get_inverse_transform(transform_name)
                inverse_color = None
                if color_map:
                    inverse_color = {v: k for k, v in color_map.items()}

                augmentation = {
                    'id': aug_id,
                    'task': transformed_task,
                    'transform_name': transform_name,
                    'color_variant': color_idx,
                    'color_map': color_map,
                    'inverse_geometric': inverse_geometric,
                    'inverse_color': inverse_color,
                    'description': f"{transform_name}_color{color_idx}"
                }

                augmentations.append(augmentation)
                aug_id += 1

                if len(augmentations) >= self.num_augmentations:
                    return augmentations

        return augmentations

    # ---------------- inverse application ----------------
    def apply_inverse_transform(
        self,
        prediction_grid: List[List[int]],
        augmentation: Dict
    ) -> List[List[int]]:
        if augmentation.get('inverse_color'):
            prediction_grid = self.permute_colors(prediction_grid, augmentation['inverse_color'])
        inverse_geometric = augmentation.get('inverse_geometric', self.identity)
        prediction_grid = inverse_geometric(prediction_grid)
        return prediction_grid

    # ---------------- consistency computation ----------------
    def compute_consistency_across_augmentations(
        self,
        original_task: Dict,
        model_or_augmentations,   # either model_predict_fn OR augmentations list
        maybe_augmentations: Optional[List[Dict]] = None
    ) -> Tuple[float, Optional[str]]:
        """
        Flexible compute_consistency function.

        Two calling patterns supported by tests:
        1) compute_consistency_across_augmentations(original_task, augmentations)
           - This uses augmentations' provided outputs (if present) and compares
             inverse-mapped outputs to the original task's test output.
        2) compute_consistency_across_augmentations(original_task, model_predict_fn, augmentations)
           - model_predict_fn(text: str) -> prediction string or grid. Uses that to evaluate
             predictions for original and augmentations.

        Returns (consistency_fraction, most_common_prediction_string_or_None)
        """
        # Determine how it was called
        if maybe_augmentations is None and isinstance(model_or_augmentations, list):
            # called as (original_task, augmentations)
            augmentations = model_or_augmentations
            model_predict_fn = None
        else:
            # called as (original_task, model_predict_fn, augmentations)
            model_predict_fn = model_or_augmentations if callable(model_or_augmentations) else None
            augmentations = maybe_augmentations if maybe_augmentations is not None else []

        # Helper local format & parser (keeps local, avoids circular import)
        def _format_augmented_task_for_llm(augmentation: Dict, format_style: str = 'compact') -> str:
            task = augmentation['task']
            if format_style == 'compact':
                lines = []
                for example in task.get('train', []):
                    lines.append('I')
                    for row in example['input']:
                        lines.append(''.join(map(str, row)))
                    lines.append('O')
                    for row in example['output']:
                        lines.append(''.join(map(str, row)))
                lines.append('I')
                if task.get('test'):
                    for row in task['test'][0]['input']:
                        lines.append(''.join(map(str, row)))
                lines.append('O')
                return '\n'.join(lines)
            else:
                prompt = "Solve task:\n"
                for i, example in enumerate(task.get('train', []), 1):
                    prompt += f"Example {i} Input:\n"
                    for row in example['input']:
                        prompt += ''.join(map(str, row)) + '\n'
                    prompt += "Output:\n"
                    for row in example['output']:
                        prompt += ''.join(map(str, row)) + '\n'
                prompt += "Test Input:\n"
                if task.get('test'):
                    for row in task['test'][0]['input']:
                        prompt += ''.join(map(str, row)) + '\n'
                prompt += "Predict the output:"
                return prompt

        def _parse_llm_output_to_grid(output):
            try:
                if output is None:
                    return None
                if isinstance(output, list):
                    return output
                s = str(output).strip()
                if s.startswith('[[') and ']]' in s:
                    return ast.literal_eval(s)
                lines = s.splitlines()
                grid = []
                for line in lines:
                    digits = [int(ch) for ch in line if ch.isdigit()]
                    if digits:
                        grid.append(digits)
                if grid and all(len(row) == len(grid[0]) for row in grid):
                    return grid
            except Exception:
                pass
            return None

        # Prepare original output in original space
        original_test_output = None
        if original_task.get('test') and 'output' in original_task['test'][0]:
            original_test_output = original_task['test'][0]['output']

        # If we have model_predict_fn, use it to get preds
        predictions_mapped_back = []  # list of (prediction_string, grid_in_original_space_or_None)

        if model_predict_fn is not None:
            # Use model to predict on original and each augmentation
            orig_text = _format_augmented_task_for_llm({'task': original_task})
            orig_pred_raw = model_predict_fn(orig_text)
            orig_grid = _parse_llm_output_to_grid(orig_pred_raw)
            # store original
            predictions_mapped_back.append(('original', orig_grid))

            for aug in augmentations:
                aug_text = _format_augmented_task_for_llm(aug)
                aug_pred_raw = model_predict_fn(aug_text)
                aug_grid = _parse_llm_output_to_grid(aug_pred_raw)
                if aug_grid is None:
                    continue
                # inverse map to original space
                try:
                    aug_grid_original = self.apply_inverse_transform(aug_grid, aug)
                except Exception:
                    aug_grid_original = aug_grid
                predictions_mapped_back.append((aug.get('description', str(aug.get('id'))), aug_grid_original))
        else:
            # No model: rely on augmentations carrying their outputs (augmentation['task']['test'][0]['output'])
            # Compare augmentations' test outputs inverse-mapped back to original test output.
            if original_test_output is None:
                # nothing to compare
                return 0.0, None
            for aug in augmentations:
                # if augmentation has 'task' and a 'test' output, use it
                try:
                    aug_test = aug.get('task', {}).get('test', [])
                    if not aug_test or 'output' not in aug_test[0]:
                        continue
                    aug_output = aug_test[0]['output']
                    # inverse-map aug_output to original space
                    try:
                        aug_grid_original = self.apply_inverse_transform(aug_output, aug)
                    except Exception:
                        aug_grid_original = aug_output
                    predictions_mapped_back.append((aug.get('description', str(aug.get('id'))), aug_grid_original))
                except Exception:
                    continue
            # also include the original task's provided output as baseline
            predictions_mapped_back.insert(0, ('original', original_test_output))

        # Now compute most common prediction (by grid equivalence) and consistency fraction
        def _grid_key(g):
            if g is None:
                return None
            return tuple(tuple(int(x) for x in row) for row in g)

        counts = {}
        valid = 0
        matches = 0
        # baseline is first element (original)
        if not predictions_mapped_back:
            return 0.0, None
        base_key = _grid_key(predictions_mapped_back[0][1])
        for name, grid in predictions_mapped_back[1:]:
            key = _grid_key(grid)
            if key is None:
                continue
            valid += 1
            if key == base_key:
                matches += 1
            counts[key] = counts.get(key, 0) + 1

        consistency = (matches / valid) if valid > 0 else 0.0
        most_common = None
        if counts:
            # find key with highest count
            most_common_key = max(counts.items(), key=lambda kv: kv[1])[0]
            # convert back to string form
            most_common = str([list(r) for r in most_common_key])

        return consistency, most_common

    # ---------------- helper equality ----------------
    @staticmethod
    def _grids_equal(g1: List[List[int]], g2: List[List[int]]) -> bool:
        if g1 is None or g2 is None:
            return False
        if len(g1) != len(g2):
            return False
        for r1, r2 in zip(g1, g2):
            if len(r1) != len(r2):
                return False
            for a, b in zip(r1, r2):
                if a != b:
                    return False
        return True


# ----------------- small local helpers used by tests -----------------
def _parse_llm_output_to_grid(output: str):
    """Lightweight parser used by augmentations module tests."""
    try:
        if output is None:
            return None
        if isinstance(output, list):
            return output
        s = str(output).strip()
        # Prefer explicit python-style grid
        if s.startswith('[[') and ']]' in s:
            return ast.literal_eval(s)
        # try lines of digits
        lines = s.splitlines()
        grid = []
        for line in lines:
            digits = [int(c) for c in line if c.isdigit()]
            if digits:
                grid.append(digits)
        if grid and all(len(row) == len(grid[0]) for row in grid):
            return grid
    except Exception:
        pass
    return None


def _format_augmented_task_for_llm(augmentation: Dict, format_style: str = 'compact') -> str:
    """
    Minimal formatter to create textual prompt expected by MITS.
    Kept local to avoid circular imports.
    """
    task = augmentation['task']
    if format_style == 'compact':
        lines = []
        for example in task.get('train', []):
            lines.append('I')
            for row in example['input']:
                lines.append(''.join(map(str, row)))
            lines.append('O')
            for row in example['output']:
                lines.append(''.join(map(str, row)))
        lines.append('I')
        # test first input
        if task.get('test'):
            for row in task['test'][0]['input']:
                lines.append(''.join(map(str, row)))
        lines.append('O')
        return '\n'.join(lines)
    else:
        prompt = "Solve task:\n"
        for i, example in enumerate(task.get('train', []), 1):
            prompt += f"Example {i} Input:\n"
            for row in example['input']:
                prompt += ''.join(map(str, row)) + '\n'
            prompt += "Output:\n"
            for row in example['output']:
                prompt += ''.join(map(str, row)) + '\n'
        prompt += "Test Input:\n"
        if task.get('test'):
            for row in task['test'][0]['input']:
                prompt += ''.join(map(str, row)) + '\n'
        prompt += "Predict the output:"
        return prompt


# ---------------- compatibility helper for tests -----------------
def prepare_augmentations_for_mits(task: Dict, augmenter: StructuralAugmenter, format_style: str = 'compact'):
    """
    Wrap generate_augmentations and attach a 'text' field used by MITS inference.
    """
    augmentations = augmenter.generate_augmentations(task)
    for aug in augmentations:
        aug['text'] = _format_augmented_task_for_llm(aug, format_style)
    return augmentations
