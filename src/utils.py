import json
import os

import matplotlib.pyplot as plt
import matplotlib


SECRETS_FILE = "secrets.json"


def load_api_key(key_name, secrets_file=SECRETS_FILE):
    """Return an API key from a local gitignored secrets file, falling back to env var."""
    if os.path.exists(secrets_file):
        with open(secrets_file, 'r') as f:
            secrets = json.load(f)
        if secrets.get(key_name):
            return secrets[key_name]
    key = os.environ.get(key_name)
    if not key:
        raise RuntimeError(
            f"{key_name} not found. Set it in {secrets_file} (gitignored; "
            f"see {secrets_file}.example) or in the {key_name} env var."
        )
    return key


# Source of truth for which datasets each task can operate on
TASK_ALLOWED_DATASETS = {
    "writing": {"ukp"},
    "improvement": {"semeval"},
    "contextualization": {"semeval"},
}

# Tasks whose outputs are bullet lists; flattened with BULLET_SENTINEL in TSV, split back at predict time
BULLET_TASKS = {"contextualization"}
BULLET_SENTINEL = " ||| "


def task_dataset_compatible(task, dataset):
    """Return True if `task` is allowed to operate on `dataset`; unknown tasks return True."""
    allowed = TASK_ALLOWED_DATASETS.get(task)
    return allowed is None or dataset in allowed


def load_prompts(prompts_file, section, task=None, variant=None):
    """Load prompts from the centralized JSON for a given section, optional task, and optional variant."""
    with open(prompts_file, 'r') as f:
        all_prompts = json.load(f)

    if section not in all_prompts:
        raise ValueError(f"Section '{section}' not found in {prompts_file}. Available: {list(all_prompts.keys())}")

    section_prompts = all_prompts[section]

    if task is not None:
        if task not in section_prompts:
            available = [k for k in section_prompts.keys() if k != "prefixes"]
            raise ValueError(f"Task '{task}' not found under section '{section}' in {prompts_file}. Available: {available}")

        task_prompts = section_prompts[task]
        prompts = task_prompts.get("prompts", [])

        if variant is not None:
            # Build per-level system prompts by prepending each steering prefix to the base system prompt
            prefixes_config = section_prompts.get("prefixes", {})
            if variant not in prefixes_config:
                raise ValueError(f"Variant '{variant}' not found under section '{section}.prefixes' in {prompts_file}. Available: {list(prefixes_config.keys())}")
            prefixes = prefixes_config[variant]

            try:
                base_system_prompt = all_prompts["transform_text"][task]["system_prompts"][0]
            except (KeyError, IndexError):
                raise ValueError(f"Base system prompt for task '{task}' not found at transform_text.{task}.system_prompts[0] in {prompts_file}.")

            system_prompts = [
                f"{prefix} {base_system_prompt}" if prefix else base_system_prompt
                for prefix in prefixes
            ]

            return {
                "system_prompts": system_prompts,
                "prompts": prompts
            }

        return task_prompts

    return section_prompts

def get_fig_dim(width, fraction=1, aspect_ratio=None):
    """Compute figure dimensions (in inches) from LaTeX width in pts."""
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    if aspect_ratio is None:
        # If not specified, set the aspect ratio equal to the Golden ratio (https://en.wikipedia.org/wiki/Golden_ratio)
        aspect_ratio = (1 + 5**.5) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in / aspect_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def latexify(font_serif='Computer Modern', mathtext_font='cm', font_size=10, small_font_size=None, usetex=True, use_defaults=False):
    """Set up matplotlib's RC params for LaTeX plotting."""

    if use_defaults:
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        plt.rcParams.update(plt.rcParamsDefault)
        return

    if small_font_size is None:
        small_font_size = font_size

    # Get available fonts
    import matplotlib.font_manager as fm
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    # Define fallback chains for common font families
    font_fallbacks = {
        'Times New Roman': ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
        'Computer Modern': ['Computer Modern', 'CMU Serif', 'Latin Modern Roman', 'cmr10'],
        'Arial': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    }

    # Try to find the best available font
    actual_font = font_serif
    if font_serif in font_fallbacks:
        # Try each fallback in order
        for fallback in font_fallbacks[font_serif]:
            if fallback in available_fonts:
                actual_font = fallback
                # Only print message if not using LaTeX and had to fall back
                if fallback != font_serif and not usetex:
                    print(f"Font '{font_serif}' not found, using '{actual_font}' instead")
                break
        else:
            if not usetex:
                print(f"Warning: Neither '{font_serif}' nor fallbacks found. Using default.")
    elif font_serif not in available_fonts and not usetex:
        print(f"Warning: Font '{font_serif}' not found. Using default.")

    params = {
        'backend': 'ps',
        'text.latex.preamble': r'\usepackage{gensymb} \usepackage{bm}',

        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'font.size': font_size,

        # Optionally set a smaller font size for legends and tick labels
        'legend.fontsize': small_font_size,
        'legend.title_fontsize': small_font_size,
        'xtick.labelsize': small_font_size,
        'ytick.labelsize': small_font_size,

        'text.usetex': usetex,
        'font.family': 'serif',
        'mathtext.fontset': mathtext_font
    }

    # Only set font.serif if not using LaTeX (LaTeX handles fonts itself)
    if not usetex:
        params['font.serif'] = actual_font

    # Fix the mathtext warning
    if not usetex and 'cm' in actual_font.lower():
        params['axes.formatter.use_mathtext'] = True

    matplotlib.rcParams.update(params)
    plt.rcParams.update(params)
