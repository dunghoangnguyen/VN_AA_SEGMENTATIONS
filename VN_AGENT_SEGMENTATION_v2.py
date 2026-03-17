# Databricks notebook source
# MAGIC %run /Repos/dung_nguyen_hoang@mfcgd.com/Utilities/Functions

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Initialization

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import Window
import pyspark.sql.functions as F
import pyspark.sql.types as T
from datetime import datetime, timedelta

pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Prepare tables

# COMMAND ----------

image_date = '2025-03-31'
image_date_sht = image_date[:7].replace('-', '')
image_year = int(image_date[:4])

cseg_path = f'/mnt/prod/Curated/VN/Master/VN_CURATED_ANALYTICS_DB/INCOME_BASED_DECILE_AGENCY/'
aseg_path = f'/mnt/lab/vn/project/cpm/datamarts/TPARDM_MTHEND/'
policy_path = f'/mnt/prod/Curated/VN/Master/VN_CURATED_REPORTS_DB/TABD_SBMT_POLS/image_date={image_date}'
out_path = '/dbfs/mnt/lab/vn/project/cpm/datamarts/AGENT_INCOME_BASED_DECILE/'

exrt_df = spark.sql(f"""
with xrt as (
select  cast(XCHNG_RATE as int) ex_rate,
        row_number() over (partition by XCHNG_RATE_TYP order by FR_EFF_DT DESC) rn
from    vn_published_cas_db.texchange_rates
where   XCHNG_RATE_TYP='U'
    and FR_CRCY_CODE='78'
    and to_date(FR_EFF_DT) <= '{image_date}'
qualify rn=1
) select ex_rate from xrt
""")
ex_rate = exrt_df.collect()[0][0]

print(image_date, image_date_sht, image_year, ex_rate)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Load and immediate tables for data preparation

# COMMAND ----------

# MAGIC %md
# MAGIC <strong> 1.2.1 Load tables</strong>

# COMMAND ----------

aseg_tmp_df = spark.read.parquet(
    aseg_path
).filter(
    F.col("image_date") == image_date
)

agent_struct_df = spark.read.csv(f"/mnt/lab/vn/project/scratch/Adhoc_Analysis/MDRT Tracking Dashboard/1. Raw Data/Structure/location_structure_03_2025.csv", header=True, inferSchema=True)
agent_struct_df.createOrReplaceTempView("agent_struct")

# Location to Province/City mapping
loc_code_city_df = spark.read.csv(f"/mnt/lab/vn/project/agency_analytics/location_city/loc_code_city.csv", header=True, inferSchema=True)
loc_code_city_df.createOrReplaceTempView("loc_code_city")

# COMMAND ----------

agt_income_df = spark.sql(f"""
with com as (
select  agt_code, pol_num, sum(nvl(comm_earn,0)) as comm_earn 
from    vn_published_cas_db.tcommission_trailers
where   to_date(cvg_eff_dt) <= '{image_date}'
    and comm_earn > 0
group by
        agt_code, pol_num
), fyp_and_comm as (
select  com.*, p_fyp.fyp, p_fyp.fyp_top_up
from    com
    left join (
        select  pol_num
                ,agt_code			
                ,nvl(sum(case when typ  = 'FYP' then amt else 0 end),0) fyp			
                ,nvl(sum(case when typ = 'FYP-TOPUP' then amt else 0 end),0) fyp_top_up
        from    vn_published_cas_db.twrk_fyp_ryp
        where   typ in ('FYP','FYP-TOPUP')
        group by
                pol_num
                ,agt_code
	) p_fyp on (com.pol_num = p_fyp.pol_num and com.agt_code = p_fyp.agt_code)
), agt_sales_dtl as (
select  a.agt_code, a.pol_num, to_date(b.pol_iss_dt) as pol_iss_dt,
        cast(nvl(a.fyp,0)+nvl(a.fyp_top_up,0) as int) as fyp,
        cast(nvl(a.comm_earn,0) as int) as comm_earn,
        b.image_date 
from    fyp_and_comm a
    inner join
        vn_published_casm_cas_snapshot_db.tpolicys b on a.pol_num = b.pol_num
    and pol_stat_cd not in ('A','N','R','X')
    and b.image_date = '{image_date}'
), agt_comm as (
select  agt_code, pol_num, pol_iss_dt, fyp, 
        sum(comm_earn) as comm_earned_total,
        sum(case when substr(to_date(pol_iss_dt),1,7) = substr(image_date,1,7) then comm_earn else 0 end) as comm_earned_lm,
        sum(case when months_between(image_date, to_date(pol_iss_dt)) <3 then comm_earn else 0 end) as comm_earned_l3m,
        sum(case when months_between(image_date, to_date(pol_iss_dt)) <6 then comm_earn else 0 end) as comm_earned_l6m,
        sum(case when months_between(image_date, to_date(pol_iss_dt)) <9 then comm_earn else 0 end) as comm_earned_l9m,
        sum(case when months_between(image_date, to_date(pol_iss_dt)) <12 then comm_earn else 0 end) as comm_earned_l12m,
        sum(case when year(pol_iss_dt) = year(image_date)-1 then comm_earn else 0 end) as comm_earned_ly,
        image_date
from    agt_sales_dtl
group by 
        agt_code, pol_num, pol_iss_dt, fyp, image_date
)
select  agt_code as agt_cd, sum(fyp) as fyp_total,
        sum(case when substr(to_date(pol_iss_dt),1,7) = substr(image_date,1,7) then fyp else 0 end) as last_mth_fyp,
        sum(case when months_between(image_date, to_date(pol_iss_dt)) <3 then fyp else 0 end) as last_3m_fyp,
        sum(case when months_between(image_date, to_date(pol_iss_dt)) <6 then fyp else 0 end) as last_6m_fyp,
        sum(case when months_between(image_date, to_date(pol_iss_dt)) <9 then fyp else 0 end) as last_9m_fyp,
        sum(case when months_between(image_date, to_date(pol_iss_dt)) <12 then fyp else 0 end) as last_12m_fyp,
        sum(case when year(pol_iss_dt) = year(image_date)-1 then fyp else 0 end) as fyp_ly,
        sum(comm_earned_total) as comm_earned_total,
        sum(comm_earned_lm) as last_mth_comm_neared,
        sum(comm_earned_l3m) as last_3m_comm_earned,
        sum(comm_earned_l6m) as last_6m_comm_earned,
        sum(comm_earned_l9m) as last_9m_comm_earned,
        sum(comm_earned_l12m) as last_12m_comm_earned,
        sum(comm_earned_ly) as ly_comm_earned,
        cast(sum(comm_earned_total)/{ex_rate}*1000 as decimal(12,2)) as total_comm_earned_usd,
        cast(sum(comm_earned_lm)/{ex_rate}*1000 as decimal(12,2)) as last_mth_comm_neared_usd,
        cast(sum(comm_earned_l3m)/{ex_rate}*1000 as decimal(12,2)) as last_3m_comm_earned_usd,
        cast(sum(comm_earned_l6m)/{ex_rate}*1000 as decimal(12,2)) as last_6m_comm_earned_usd,
        cast(sum(comm_earned_l9m)/{ex_rate}*1000 as decimal(12,2)) as last_9m_comm_earned_usd,
        cast(sum(comm_earned_l12m)/{ex_rate}*1000 as decimal(12,2)) as last_12m_comm_earned_usd,
        cast(sum(comm_earned_ly)/{ex_rate}*1000 as decimal(12,2)) as ly_comm_earned_usd,
        image_date
from    agt_comm
group by
        agt_code, image_date
""")



# COMMAND ----------

# DBTITLE 1,Merge agent income to TPARDM
aseg_df = aseg_tmp_df.join(
    agt_income_df, on=["agt_cd","image_date"], how="left"
).fillna(
    {"fyp_total": 0, "last_mth_fyp": 0, "last_3m_fyp": 0, "last_6m_fyp": 0, "last_9m_fyp": 0, "last_12m_fyp": 0, "fyp_ly": 0,
     "comm_earned_total": 0, "last_mth_comm_neared": 0, "last_3m_comm_earned": 0, "last_6m_comm_earned": 0, "last_9m_comm_earned": 0, 
     "last_12m_comm_earned": 0, "ly_comm_earned": 0, "total_comm_earned_usd": 0, "last_mth_comm_neared_usd": 0, "last_3m_comm_earned_usd": 0, 
     "last_6m_comm_earned_usd": 0, "last_9m_comm_earned_usd": 0, "last_12m_comm_earned_usd": 0, "ly_comm_earned_usd": 0}
)

aseg_df = aseg_df.withColumn(
    "mthly_incm",
    F.when(F.col("comm_earned_total")>0, F.round(F.col("comm_earned_total") / F.col("tenure_mth"), 0))
    .otherwise(0)
    .cast("int")
).withColumn(
    "last_12m_mthly_incm",
    F.when(F.col("last_12m_comm_earned")>0, F.round(F.col("last_12m_comm_earned") / 12, 0))
    .otherwise(0)
    .cast("int")
).withColumn(
    "mthly_incm_usd",
    F.when(F.col("total_comm_earned_usd")>0, F.col("total_comm_earned_usd") / F.col("tenure_mth"))
    .otherwise(0)
    .cast("decimal(12,2)")
).withColumn(
    "last_12m_mthly_incm_usd",
    F.when(F.col("last_12m_comm_earned_usd")>0, F.col("last_12m_comm_earned_usd") / 12)
    .otherwise(0)
    .cast("decimal(12,2)")
).withColumn(
    "income_class",
    F.when(F.col("mthly_incm") == 0, "G. No Income")
    .when(F.col("mthly_incm") < 5000, "F. Under 5M")
    .when((F.col("mthly_incm") >= 5000) & (F.col("mthly_incm") <= 7500), "E. 5M - 7.5M")
    .when((F.col("mthly_incm") > 7500) & (F.col("mthly_incm") <= 10000), "D. 7.5M - 10M")
    .when((F.col("mthly_incm") > 10000) & (F.col("mthly_incm") <= 15000), "C. 10M - 15M")
    .when((F.col("mthly_incm") > 15000) & (F.col("mthly_incm") <= 20000), "B. 15M - 20M")
    .when(F.col("mthly_incm") > 20000, "A. Above 20M")
    .otherwise("G. No Income")
).withColumn(
    "last_12m_income_class",
    F.when(F.col("last_12m_mthly_incm") == 0, "G. No Income")
    .when(F.col("last_12m_mthly_incm") < 5000, "F. Under 5M")
    .when((F.col("last_12m_mthly_incm") >= 5000) & (F.col("last_12m_mthly_incm") <= 7500), "E. 5M - 7.5M")
    .when((F.col("last_12m_mthly_incm") > 7500) & (F.col("last_12m_mthly_incm") <= 10000), "D. 7.5M - 10M")
    .when((F.col("last_12m_mthly_incm") > 10000) & (F.col("last_12m_mthly_incm") <= 15000), "C. 10M - 15M")
    .when((F.col("last_12m_mthly_incm") > 15000) & (F.col("last_12m_mthly_incm") <= 20000), "B. 15M - 20M")
    .when(F.col("last_12m_mthly_incm") > 20000, "A. Above 20M")
    .otherwise("G. No Income")
).drop("index","agt_nm","lps_pol_list")

# COMMAND ----------

aseg_df.createOrReplaceTempView("all_agents")
all_agent_level = spark.sql("""
select  a.*,
        case tier
            when 'TOT'  then 1
            when 'COT'  then 2
            when 'MDRT' then 3
            when 'Platinum' then 4
            when 'Gold' then 5
            when 'Silver' then 6
            when 'Unranked' then 7
        end as tier_rank,
        case when a.last_9m_pol > 0 then 0 else 1 end as sa_status,
        ntile(10) over (order by a.mthly_incm_usd desc) as income_decile ,
        ntile(10) over (order by a.last_12m_mthly_incm_usd desc) as income_decile_last_12m,
        floor(months_between(image_date, to_date(b.birth_dt)) / 12) as agent_age,
        concat_ws('-',c.rh_agt_code,c.rh_name) as psm_lv_1_name,
        concat_ws('-',c.rh_1_agt_code,c.rh_1_name) as psm_lv_2_name,
        b.loc_cd,
        d.city,
        d.city_vnese,
        d.region,
        d.channel
from    all_agents a inner join
        vn_curated_datamart_db.tagtdm_daily b on a.agt_cd = b.agt_code left join
        agent_struct c on b.loc_cd = c.loc_code left join
        loc_code_city d on b.loc_cd = d.loc_code
""")

all_agent_level = all_agent_level.withColumn(
        "age_band",
        F.when((F.col("agent_age") >= 18) & (F.col("agent_age") <= 29), "1. 18 - 29")
        .when((F.col("agent_age") >= 30) & (F.col("agent_age") <= 44), "2. 30 - 44")
        .when((F.col("agent_age") >= 45) & (F.col("agent_age") <= 54), "2. 45 - 54")
        .when(F.col("agent_age") > 54, "4. 55+")
        .otherwise("0. Under 18")
)

# agent_count = all_agent_level.count()
# print("Agent count:",agent_count)

# COMMAND ----------

all_agent_level.groupBy(
    "last_12m_income_class"
).agg(
    F.count(F.col("agt_cd")).alias("agent_count"),
    #F.mean(F.col("mthly_incm")).cast("int").alias("mthly_income"),
    F.mean(F.col("last_12m_mthly_incm")).cast("int").alias("mthly_income_last_12m")
).orderBy(
    "last_12m_income_class"
).display()

# COMMAND ----------

all_agent_level.filter(
    F.col("sa_status") == 0
).groupBy(
    "region",
    "city",
    "last_12m_income_class",
    "age_band"
).agg(
    F.count(F.col("agt_cd")).alias("agent_count"),
    #F.mean(F.col("mthly_incm")).cast("int").alias("mthly_income"),
    F.mean(F.col("last_12m_mthly_incm")).cast("int").alias("avg_mthly_income_last_12m")
).orderBy(
    "last_12m_income_class", "age_band"
).display()

# COMMAND ----------

result = all_agent_level.filter(
    F.col("sa_status") == 0
).withColumn(
    "tier",
    F.when(F.col("tier").isin("TOT","COT"), "MDRT")
    .otherwise(F.col("tier"))
).groupBy(
    ("tier")
).agg(
    F.count(F.col("agt_cd")).alias("count"),
    F.mean(F.col("last_12m_mthly_incm")).alias("mthly_income_last_12m"),
    F.mean(F.col("mthly_incm")).alias("mthly_income")
).orderBy(
    F.desc("mthly_income")
)

result.limit(10).toPandas()

# COMMAND ----------

all_agent_level.toPandas().to_csv(f"{out_path}CSV_files/agent_income_based_{image_date_sht}.csv", index=False, header=True, encoding='utf-8-sig')

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Load Customer segments with Income and VIP ranks

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Group all metrics for slicers
# MAGIC <strong>Agent Base – 54k</strong><br>
# MAGIC •	Active/ Inactive<br>
# MAGIC •	Active – 0-3M, 3-6M, 6-9M, >9M<br>
# MAGIC •	Location/ Branch/ SM<br>
# MAGIC •	APE under agent<br>
# MAGIC •	No. of customers<br>
# MAGIC •	No. of policies<br>
# MAGIC •	MOB<br>
# MAGIC •	MDRT/ Manupro status<br>
# MAGIC •	APE required to reach next level (Gold to Plat, Silver to Gold etc.)<br>

# COMMAND ----------

# MAGIC %md
# MAGIC ###2.3 Store detailed data for future sizing

# COMMAND ----------

cseg_tmp_df = spark.sql(f"""
with po_ape as (
select  po_num, cast(sum(nvl(base_ape,0)+nvl(rid_ape,0)) as int) as tot_ape
from    vn_curated_datamart_db.tpolidm_daily
where   POL_STAT_CD in ('1','3','5')
group by po_num
), all_cus as (
select    distinct
          pol.WA_CODE as agt_code,
          pol.PO_NUM as po_num,
          cast(cus.MTHLY_INCM as int) as mthly_incm,
          nvl(vip.VIP_TYP_DESC,'Non-VIP') as vip_type,
          cast(sum(nvl(pol.base_ape,0)+nvl(pol.rid_ape,0)) as int) as tot_ape,
          case when cus.MTHLY_INCM is null or cus.MTHLY_INCM=0 then 1 else 0 end missing_income_ind
from      vn_curated_datamart_db.tcustdm_daily cus
  inner join
          vn_curated_datamart_db.tpolidm_daily pol on cus.CLI_NUM = pol.PO_NUM
  left join
          vn_published_cas_db.twrk_client_ape vip on cus.CLI_NUM = vip.CLI_NUM
where     pol.POL_STAT_CD not in ('8','A','R','X')
group by  agt_code, po_num, cus.MTHLY_INCM, vip.VIP_TYP_DESC
), revised_incm as (
select  agt_code, all.po_num, vip_type,
        case when missing_income_ind=1 then cast(nvl(ape.tot_ape,0) * 2 as int) else mthly_incm end as mthly_incm,
        missing_income_ind
from    all_cus all
  left join
        po_ape ape on all.po_num = ape.po_num
--where   tot_ape > 0
)
select    agt_code as agt_cd, po_num, vip_type, mthly_incm, 
        missing_income_ind,
          case when mthly_incm < 10000 then "F. Under 10M"
               when mthly_incm >= 10000 and mthly_incm <= 15000 then "E. 10M - 15M"
               when mthly_incm > 15000 and mthly_incm <= 20000 then "D. 15M - 20M"
               when mthly_incm > 20000 and mthly_incm <= 30000 then "C. 20M - 30M"
               when mthly_incm > 30000 and mthly_incm <= 40000 then "B. 30M - 40M"
               else "A. Above 40M"
          end as vip_income_class
from      revised_incm
""")

cus_count = cseg_tmp_df.count()
print(cus_count)
#cseg_tmp_df.limit(10).display()

# COMMAND ----------

agt_cseg_df = cseg_tmp_df.groupBy(
    "agt_cd"
).agg(
    F.sum(F.when(F.col("vip_type") == "SUPER VIP", F.lit(1)).otherwise(0)).alias("a_super_vip_count"),
    F.sum(F.when(F.col("vip_type") == "VIP Platinum", F.lit(1)).otherwise(0)).alias("a_plat_vip_count"),
    F.sum(F.when(F.col("vip_type") == "VIP Gold", F.lit(1)).otherwise(0)).alias("a_gold_vip_count"),
    F.sum(F.when(F.col("vip_type") == "VIP Silver", F.lit(1)).otherwise(0)).alias("a_silver_vip_count"),
    F.sum(F.when(F.col("vip_type") == "Non-VIP", F.lit(1)).otherwise(0)).alias("a_nonvip_count"),
    F.sum(F.when(F.col("vip_income_class") == "A. Above 40M", F.lit(1)).otherwise(0)).alias("a_class_A_count"),
    F.sum(F.when(F.col("vip_income_class") == "B. 30M - 40M", F.lit(1)).otherwise(0)).alias("a_class_B_count"),
    F.sum(F.when(F.col("vip_income_class") == "C. 20M - 30M", F.lit(1)).otherwise(0)).alias("a_class_C_count"),
    F.sum(F.when(F.col("vip_income_class") == "D. 15M - 20M", F.lit(1)).otherwise(0)).alias("a_class_D_count"),
    F.sum(F.when(F.col("vip_income_class") == "E. 10M - 15M", F.lit(1)).otherwise(0)).alias("a_class_E_count"),
    F.sum(F.when(F.col("vip_income_class") == "F. Under 10M", F.lit(1)).otherwise(0)).alias("a_class_F_count")
)

#print(agt_cseg_df.count())

# COMMAND ----------

selected_cols = ["agt_cd", "tier", "last_12m_income_class", "tier_rank", "sa_status", "income_decile_last_12m", "psm_lv_1_name",
                 "psm_lv_2_name", "loc_cd", "city", "city_vnese", "region", "channel", "age_band"]

all_agent_level = F.broadcast(all_agent_level)

all_cus_level = agt_cseg_df.join(
    all_agent_level.select(*selected_cols), on="agt_cd", how="inner"
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Analysis

# COMMAND ----------

# Load cseg data
result = all_cus_level.groupBy(
    "sa_status",
    "tier",
    "region",
    "city",
    "last_12m_income_class",
    "age_band"
).agg(
    F.count("agt_cd").alias("agent_count"),
    F.sum("a_super_vip_count").alias("a_super_vip_count"),
    F.sum("a_plat_vip_count").alias("a_plat_vip_count"),
    F.sum("a_gold_vip_count").alias("a_gold_vip_count"),
    F.sum("a_silver_vip_count").alias("a_silver_vip_count"),
    F.sum("a_nonvip_count").alias("a_nonvip_count"),
    F.sum("a_class_A_count").alias("a_class_A_count"),
    F.sum("a_class_B_count").alias("a_class_B_count"),
    F.sum("a_class_C_count").alias("a_class_C_count"),
    F.sum("a_class_D_count").alias("a_class_D_count"),
    F.sum("a_class_E_count").alias("a_class_E_count"),
    F.sum("a_class_F_count").alias("a_class_F_count")
)

result.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Customer Analysis
