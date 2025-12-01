# Cat Chase Gym Environment

This project provides a simple OpenAI Gym environment where an agent (green circle) tries to catch a cat (sprite) on an 8x8 grid. The cat moves randomly after every agent move.

Files:
- `cat_env.py`: The Gym environment implementation (class `CatChaseEnv`).
- `example.py`: Simple runner that plays a random policy and renders the environment.
- `play.py`: Interactive version where you control the agent with arrow keys.
- `images/peekaboo.png`: Cat sprite used when rendering (already in the workspace).
- `requirements.txt`: Dependencies.

Quick run (PowerShell):

```powershell
# Install dependencies
python -m pip install -r requirements.txt

# Watch random agent play
python example.py

# Play yourself using arrow keys
python play.py
```

Quick run (macOS / zsh):

```bash
# Activate the project's virtual environment (from repository root)
cd "/Users/francescacatolico/Downloads/release 2"
source .venv/bin/activate

# Install Python dependencies (if not already installed lol kasi nagkamali ako here lots)
python -m pip install --upgrade pip setuptools wheel
pip install -r catbot/requirements.txt

# Watch random agent play
python catbot/example.py

# Play yourself (interactive window). Click the game window before using arrow keys.
python catbot/play.py --cat mittens

# Run training (5000 episodes) for a cat:
python catbot/bot.py --cat paotsin

# Play a saved Q-table (visual demo):
python - <<'PY'
import pickle
from cat_env import make_env
from utility import play_q_table

with open('catbot/artifacts/q_peekaboo_perfect_run_5.pkl','rb') as f:
	q = pickle.load(f)

env = make_env(cat_type='peekaboo')
play_q_table(env, q, move_delay=0.05, max_steps=200, window_title='peekaboo demo')
PY
```

Controls (when using play.py):
- ↑ Move up
- ↓ Move down
- ← Move left
- → Move right
- Q Quit game

If `pygame` fails to open a window under certain remote or headless environments, try running locally with an active display.

## Francesca Catolico's notes !! :D

- I implemented Q-learning in `training.py` and ran headless training locally (temporary dev patch). For submission I reverted the dev-only changes so the repository matches the original project constraints (only `TrainerCat` allowed to be modified).
- Trained Q-tables (produced during local runs) are saved in the project folder as `q_batmeow.pkl`, `q_mittens.pkl`, `q_paotsin.pkl`, `q_peekaboo.pkl`, and `q_squiddyboi.pkl`.

To reproduce training or run the project with rendering you need to install system SDL2 libs and `pygame`:

```bash
# macOS (Homebrew):
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf pkg-config
sudo xcode-select --install  # if you need Command Line Tools

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If you want to run training headless (without installing `pygame`) a small dev-only patch can be applied to make rendering optional; I used that during development to produce the Q-tables above. 
