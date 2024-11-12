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
from src.email_utils import (
    is_valid_email,
    email_str_to_df,
    extract_email_groups,
)
```

## Test list

```python
df_test = pd.DataFrame(
    columns=["email", "name", "trigger", "info"],
    data=[
        ["tristan.downing@un.org", "TEST_NAME", "to", "to"],
        ["downing.tristan@gmail.com", "TEST_NAME", "to", None],
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
raw_emails_df = blob.load_csv_from_blob(
    f"{blob.PROJECT_PREFIX}/raw/20240911_Fiji AA_Trigger distribution lists(INPUT).csv"
)
```

```python
email_groups = extract_email_groups(raw_emails_df)
```

```python
email_groups.get("info_cc")
```

```python
trigger_to_df = pd.DataFrame(email_groups.get("trigger_to"))
trigger_to_df["trigger"] = "to"
```

```python
dfs_email_type = []
for email_type in ["trigger", "info"]:
    dfs = []
    for to_cc in ["to", "cc"]:
        group_name = f"{email_type}_{to_cc}"
        df_in = pd.DataFrame(
            data=email_groups.get(group_name), columns=["email"]
        )
        df_in[email_type] = to_cc
        dfs.append(df_in)
    df_email_type = pd.concat(dfs, ignore_index=True)
    df_email_type = df_email_type[~df_email_type.duplicated(subset="email")]
    dfs_email_type.append(df_email_type)

df_all = dfs_email_type[0].merge(dfs_email_type[1], how="outer")
df_all = df_all.fillna("")
df_all["name"] = df_all["email"]
```

```python
df_all
```

```python
print("invalid emails: ")
display(df_all[~df_all["email"].apply(is_valid_email)])
```

```python
blob_name = f"{blob.PROJECT_PREFIX}/email/distribution_list.csv"
blob.upload_csv_to_blob(blob_name, df_all)
df_all
```

### String method (include names)

<!-- markdownlint-disable MD013 -->
#### Trigger `to`

```python
trigger_to_str = "Dirk Wagener <dirk.wagener@un.org>; Agus Wandi <agus.wandi@un.org>; Alumita Masitabua <alumita.masitabua@un.org>; RC Office Fiji <rco.fiji@un.org>"
trigger_to_df = email_str_to_df(trigger_to_str)
trigger_to_df["trigger"] = "to"
```

#### Trigger `cc`

```python
trigger_cc_ocha_str = "Olga Prorovskaya <prorovskaya@un.org>; Hanna Paulose <hanna.paulose@un.org>; Sara Manni <sara.manni@un.org>; Valerie Broudic <valerie.broudic@un.org>; Katalaine Duaibe <katalaine.duaibe@un.org>; Tristan Downing <tristan.downing@un.org>; Daniel Pfister <pfisterd@un.org>; Yakubu Alhassan <yakubu.alhassan@un.org>; Daniel Gilman <gilmand@un.org>; Julia Wittig <wittigj@un.org>; OCHA-cerf <cerf@un.org>; Regina Omlor <regina.omlor@un.org>; Michael Jensen <jensen7@un.org>; Nicolas Rost <rostn@un.org>; Madoka Koide <koide@un.org>; Mereoni Ketewai <mereoni.ketewai@un.org>"
trigger_cc_ndmo_str = "vasiti.soko@gmail.com <vasiti.soko@gmail.com>; Litiana Bainimarama Bainimarama [GOVNET GOV] <lbainimarama@govnet.gov.fj>; napsboseiwaqa <napsboseiwaqa@gmail.com>; Prishika Nadan <prishinadan@gmail.com>; neocfiji@gmail.com <neocfiji@gmail.com>; Lnaidoleca <lnaidoleca@gmail.com>"
trigger_cc_unfpa_str = "Iori Kato [UNFPA] <kato@unfpa.org>; Mateen Shaheen [UNFPA] <shaheen@unfpa.org>; Ana Maria [UNFPA] <leal@unfpa.org>; Abdul Hussain <abdhussain@unfpa.org>; Olana Naigulevu [UNFPA] <naigulevu@unfpa.org>"
trigger_cc_iom_str = "BIDDER Matthew Aaron Mark <mbidder@iom.int>; Solomon Kantha [IOM] <skantha@iom.int>; Karishma Devi [IOM] <kdevi@iom.int>; RO Bangkok - Emergency and Post-Crisis Team <ROBangkokEPC@iom.int>; NAM Yunae <ynam@iom.int>"
trigger_cc_unicef_str = "Roshni Basu� [UNICEF] <rbasu@unicef.org>; Diego Fernando Lopez <dflopez@unicef.org>; Ali Safarnejad [UNICEF] <asafarnejad@unicef.org>; Halitesh Datt <hdatt@unicef.org>; Semiti Temo <stemo@unicef.org>; Ronesh Prasad [UNICEF] <roprasad@unicef.org>; Kencho Namgyal <knamgyal@unicef.org>"
trigger_cc_who_str = "JACOBS, Mark Andrew <jacobsma@who.int>; MAHMOUD, Nuha <hamidn@who.int>; GILMORE, Chandra <gilmorec@who.int>; KABETHYMER, Biniam Getachew <kabethymerb@who.int>; DE SOUZA, Jeff Brian Delali <deje@who.int>"
trigger_cc_wfp_str = "Alpha Bah [WFP] <alpha.bah@wfp.org>; Emma CONLAN [WFP] <emma.conlan@wfp.org>; Philippe BREWSTER <philippe.brewster@wfp.org>; Nitesh CHAND <nitesh.chand@wfp.org>; Alexander THOMAS <alexander.thomas@wfp.org>; Adam MCVIE <adam.mcvie@wfp.org>; Jorge DIAZ <jorge.diaz@wfp.org>; Tasneem ALBAGIR <tasneem.albagir@wfp.org>; Pratika DEVI <pratika.devi@wfp.org>; Saidamon BODAMAEV <saidamon.bodamaev@wfp.org>; Daniel LONGHURST <daniel.longhurst@wfp.org>; Urbe SECADES <urbe.secades@wfp.org>; Bronwyn Healy-Aarons [WFP] <bronwyn.healy-aarons@wfp.org>; Diego FLORES <diego.flores@wfp.org>"
trigger_cc_str = "; ".join(
    [
        trigger_cc_ocha_str,
        trigger_cc_ndmo_str,
        trigger_cc_unfpa_str,
        trigger_cc_iom_str,
        trigger_cc_unicef_str,
        trigger_cc_who_str,
        trigger_cc_wfp_str,
    ]
)
trigger_cc_df = email_str_to_df(trigger_cc_str)
trigger_cc_df["trigger"] = "cc"
```

#### Info `to`

```python
info_to_rc_str = "Agus Wandi <agus.wandi@un.org>; Alumita Masitabua <alumita.masitabua@un.org>"
info_to_str = "; ".join([info_to_rc_str])
info_to_df = email_str_to_df(info_to_str)
info_to_df["info"] = "to"
```

```python
df_all_str = pd.concat(
    [trigger_to_df, trigger_cc_df],
).merge(pd.concat([info_to_df]), how="outer")
df_all_str = df_all_str.fillna(value="")
df_all_str
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
