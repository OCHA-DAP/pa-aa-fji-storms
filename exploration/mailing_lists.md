---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: pa-aa-fji-storms
    language: python
    name: pa-aa-fji-storms
---

# Mailing lists

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd

from src import blob
from src.email_utils import is_valid_email
```

## Test list

```python
df_test = pd.DataFrame(
    columns=["email", "name", "trigger", "info"],
    data=[
        ["tristan.downing@un.org", "TEST_NAME", "to", "to"],
        ["downing.tristan@gmail.com", "TEST_NAME", "to", None],
    ]
    * 51,
)
print("invalid emails: ")
display(df_test[~df_test["email"].apply(is_valid_email)])
blob_name = f"{blob.PROJECT_PREFIX}/email/test_distribution_list.csv"
blob.upload_csv_to_blob(blob_name, df_test)
df_test
```

## SimEx list

```python
df_simex = pd.DataFrame(
    columns=["email", "name", "trigger", "info"],
    data=[
        ["tristan.downing@un.org", "", "to", "to"],
        ["downing.tristan@gmail.com", "TEST_NAME", "to", "cc"],
    ],
)
print("invalid emails: ")
display(df_simex[~df_simex["email"].apply(is_valid_email)])
blob_name = f"{blob.PROJECT_PREFIX}/email/simex_distribution_list.csv"
blob.upload_csv_to_blob(blob_name, df_simex)
df_simex
```

## Actual list

```python
df_actual = pd.DataFrame(
    columns=["email", "name", "trigger", "info"],
    data=[
        ["tristan.downing@un.org", "TEST_NAME", "to", "to"],
        ["downing.tristan@gmail.com", "TEST_NAME", "to", "to"],
    ],
)
print("invalid emails: ")
display(df_actual[~df_actual["email"].apply(is_valid_email)])
df_actual
```

```python
blob_name = f"{blob.PROJECT_PREFIX}/email/distribution_list.csv"
blob.upload_csv_to_blob(blob_name, df_actual)
df_actual
```

```python

```
