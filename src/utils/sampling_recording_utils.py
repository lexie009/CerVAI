import pandas as pd
import os
from typing import List

def record_sampling_info(
    csv_path: str,
    selected_image_names: List[str],
    round_name: str,
    strategy_name: str,
    model_name: str,
    set_tag: str = 'train'
) -> None:
    """
    Update/create CSV with sampling info, allowing multiple strategies per image.

    Args:
        csv_path: Path to the metadata CSV file.
        selected_image_names: List of image file names selected this round.
        round_name: e.g. 'round0'
        strategy_name: e.g. 'Entropy', 'Borda'
        set_tag: usually 'train'
    """
    selected_image_names = [name.strip() for name in selected_image_names]

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=[
            'new_image_name', 'new_mask_name', 'set',
            'swede_category', 'strategies','model_name',
        ])

    # Add missing columns if needed
    if 'model' not in df.columns:
        df['model'] = None
    if round_name not in df.columns:
        df[round_name] = False

    # Add round column if missing
    if round_name not in df.columns:
        df[round_name] = False

    for img_name in selected_image_names:
        mask_name = img_name.replace('.png', '_mask.png')
        existing = df['new_image_name'] == img_name

        if existing.any():
            idx = df[existing].index[0]

            # Mark current round
            df.at[idx, round_name] = True

            # Append strategy (avoid duplicates)
            prev = df.at[idx, 'strategies'] if pd.notna(df.at[idx, 'strategies']) else ''
            prev_list = set([s.strip() for s in prev.split(',') if s])
            prev_list.add(strategy_name)
            df.at[idx, 'strategies'] = ','.join(sorted(prev_list))

        else:
            new_row = {
                'new_image_name': img_name,
                'new_mask_name': mask_name,
                'set': set_tag,
                'swede_category': None,
                'strategies': strategy_name,
                'model_name': model_name,
                round_name: True
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"[INFO] Recorded {len(selected_image_names)} images for {round_name} ({strategy_name})")
