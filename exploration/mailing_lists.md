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
        # ["downing.tristan@gmail.com", "TEST_NAME", "to", None],
    ],
)
print("invalid emails: ")
display(df_test[~df_test["email"].apply(is_valid_email)])
blob_name = f"{blob.PROJECT_PREFIX}/email/test_distribution_list.csv"
blob.upload_csv_to_blob(blob_name, df_test)
df_test
```

## Actual list

```python
blob_name = f"{blob.PROJECT_PREFIX}/raw/"
```

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

## SimEx list

```python
# without names
df_simex = pd.DataFrame(
    columns=["email", "name", "trigger", "info"],
    data=[
        ["leal@unfpa.org", "", "to", "to"],
        ["kunal.lal@un.org", "", "to", "to"],
        ["pravneil.c@habitatfiji.org.fj", "", "to", "to"],
        ["naigulevu@unfpa.org", "", "to", "to"],
        ["inyoung.jang@fao.org", "", "to", "to"],
        ["abdhussain@unfpa.org", "", "to", "to"],
        ["losalini_nalawa@habitatfiji.org.fj", "", "to", "to"],
        ["raziya.s@habitatfiji.org.fj", "", "to", "to"],
        ["sangita.k@habitatfiji.org.fj", "", "to", "to"],
        ["SCagilaba@unicef.org", "", "to", "to"],
        ["ynam@iom.int", "", "to", "to"],
        ["Gilmorec@who.int", "", "to", "to"],
        ["hrasheed@unfpa.org", "", "to", "to"],
        ["alvina.karan@wfp.org", "", "to", "to"],
        ["nazgul.borkosheva@un.org", "", "to", "to"],
        ["hdatt@unicef.org", "", "to", "to"],
        ["dflopez@unicef.org", "", "to", "to"],
        ["prishinadan@gmail.com", "", "to", "to"],
        ["mereoni.ketewai@un.org", "", "to", "to"],
        ["valerie.broudic@un.org", "", "to", "to"],
        ["nemaia.koto@un.org", "", "to", "to"],
        ["kolosa.matebalavu@unwomen.org", "", "to", "to"],
        ["sara.manni@un.org", "Sara Manni", "cc", "cc"],
        ["tristan.downing@un.org", "Tristan Downing", "cc", "cc"],
        # ["downing.tristan@gmail.com", "TEST_NAME", "to", "cc"],
    ],
)
print("invalid emails: ")
display(df_simex[~df_simex["email"].apply(is_valid_email)])
df_simex
```

```python
# with names
df_simex = pd.DataFrame(
    columns=["email", "name", "trigger", "info"],
    data=[
        ["leal@unfpa.org", "Ana Maria Leal", "to", "to"],
        ["kunal.lal@un.org", "Kunal Lal", "to", "to"],
        ["pravneil.c@habitatfiji.org.fj", "Pravneil Chand", "to", "to"],
        ["naigulevu@unfpa.org", "Olana Naigulevu", "to", "to"],
        ["inyoung.jang@fao.org", "Inyoung Jang", "to", "to"],
        ["abdhussain@unfpa.org", "Abdul Raiyaz Hussain", "to", "to"],
        ["losalini_nalawa@habitatfiji.org.fj", "Losalini Nalawa", "to", "to"],
        ["raziya.s@habitatfiji.org.fj", "Raziya Saiyed", "to", "to"],
        ["sangita.k@habitatfiji.org.fj", "Sangita Kumar", "to", "to"],
        ["SCagilaba@unicef.org", "Seruwaia Cagilaba", "to", "to"],
        ["ynam@iom.int", "Yunae Nam", "to", "to"],
        ["Gilmorec@who.int", "Chandra Gilmore", "to", "to"],
        ["hrasheed@unfpa.org", "Haider Rasheed", "to", "to"],
        ["alvina.karan@wfp.org", "Alvina Karan", "to", "to"],
        ["nazgul.borkosheva@un.org", "Nazg Borkosheva", "to", "to"],
        ["hdatt@unicef.org", "Halitesh Datt", "to", "to"],
        ["dflopez@unicef.org", "", "to", "to"],
        ["prishinadan@gmail.com", "", "to", "to"],
        ["mereoni.ketewai@un.org", "", "to", "to"],
        ["valerie.broudic@un.org", "", "to", "to"],
        ["nemaia.koto@un.org", "", "to", "to"],
        ["kolosa.matebalavu@unwomen.org", "", "to", "to"],
        ["nete.logavatu@redcross.com.fj", "", "to", "to"],
        # ["SCagilaba@unicef.org", "Seruwaia Cagilaba", "to", "to"],
        ["sara.manni@un.org", "Sara Manni", "cc", "cc"],
        ["tristan.downing@un.org", "Tristan Downing", "cc", "cc"],
        # ["downing.tristan@gmail.com", "TEST_NAME", "to", "cc"],
    ],
)
print("invalid emails: ")
display(df_simex[~df_simex["email"].apply(is_valid_email)])
df_simex
```

```python
blob_name = f"{blob.PROJECT_PREFIX}/email/simex_distribution_list.csv"
blob.upload_csv_to_blob(blob_name, df_simex)
df_simex
```

```python

```
