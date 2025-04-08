import pandas as pd


def preprocess_data(grades, weightage, out_of_marks):
    import pandas as pd

    df = pd.DataFrame(grades)
    df.fillna(0, inplace=True)

    # Convert to numeric (in case of strings)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Group columns by component type (e.g., Quiz1, Quiz2 → Quiz)
    entity_groups = {}
    for col in df.columns:
        for entity in weightage:
            col_clean = col.replace(" ", "").lower()
            entity_clean = entity.replace(" ", "").lower()
            if col_clean.startswith(entity_clean):
                entity_groups.setdefault(entity, []).append(col)

    # Normalize: percentage → weighted percentage
    for entity, columns in entity_groups.items():
        n = len(columns)
        weight_per_column = weightage[entity] / n
        out_of = out_of_marks.get(entity, 1)

        for col in columns:
            # Convert raw score → percentage of `out_of` → weighted
            df[col] = ((df[col] / out_of) * weight_per_column).round(2)

    return df

