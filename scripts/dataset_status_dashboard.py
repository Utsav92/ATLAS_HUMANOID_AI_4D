"""Dataset Status Dashboard CLI for ATLAS HUMANOID AI 4D.
Displays the mandatory dataset lifecycle progression for all 13 sources + ATLAS Language-to-Action.
"""

from atlas.datasets.registry import DatasetRegistry


def main():
    reg = DatasetRegistry()
    print(reg.get_status_dashboard())


if __name__ == "__main__":
    main()
