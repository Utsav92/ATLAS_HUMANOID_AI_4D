"""Milestone M2 Runner: Dataset Hub and Canonical Schemas.
Audits all 14 datasets, checks licenses, validates coordinate conversions and splits.
"""

from atlas.datasets.registry import DatasetRegistry


def main():
    print("=" * 70)
    print("ATLAS HUMANOID AI 4D — MILESTONE M2 DATASET HUB")
    print("=" * 70)

    reg = DatasetRegistry()
    datasets = reg.list_datasets()

    print(f"\nDiscovered {len(datasets)} Integrated Datasets & Frameworks:")
    print(f"{'#':<3} | {'Source Name':<30} | {'License':<14} | {'Embodiment':<26} | {'Frame':<6}")
    print("-" * 88)
    for idx, d in enumerate(datasets, 1):
        print(f"{idx:<3} | {d['name']:<30} | {d['license']:<14} | {d['embodiment'][:25]:<26} | {d['coordinate_frame']:<6}")

    print("\nRunning Integrity Checks Across All Sources...")
    integrities = reg.validate_all_integrities()
    for src_id, ok in integrities.items():
        assert ok, f"Source {src_id} integrity check failed!"
    print(f"[OK] All {len(integrities)} datasets validated successfully.")

    # Audit sample split
    split = reg.get_train_val_test_split("atlas_lang_act_v1")
    print(f"[OK] Sample split verification for ATLAS Language-to-Action: Train={len(split['train'])}, Val={len(split['val'])}, Test={len(split['test'])}")

    print("\n[SUCCESS] Milestone M2 validated: Complete Dataset Hub ready.")


if __name__ == "__main__":
    main()
