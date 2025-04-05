import pandas as pd

def preprocess_data(grades, weightage):
    """
    Preprocess the student grades by normalizing based on weightage.
    Missing values are replaced with 0.
    """
    df = pd.DataFrame(grades)
    df.fillna(0, inplace=True)

    # Convert all values to numeric (string to int/float)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Group columns by entity (e.g., Quiz1, Quiz2 → Quiz)
    entity_groups = {}
    for col in df.columns:
        for entity in weightage:
            col_clean = col.replace(" ", "").lower()
            entity_clean = entity.replace(" ", "").lower()
            if col_clean.startswith(entity_clean):
                entity_groups.setdefault(entity, []).append(col)

    # Normalize each grouped column to percentage
    for entity, columns in entity_groups.items():
        for col in columns:
            max_score = df[col].max() if df[col].max() > 0 else 1
            df[col] = ((df[col] / max_score) * 100).round(2)

    return df
