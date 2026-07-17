import sys

from guitartab.cli import main
from guitartab.env import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    sys.exit(main())
