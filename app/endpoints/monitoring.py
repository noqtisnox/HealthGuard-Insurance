"""Monitoring endpoints."""


import sqlite3
import pandas as pd
import os

from fastapi import APIRouter
from fastapi.responses import FileResponse
from ..services.evidently_reports import (
    generate_data_summary_report,
    generate_feature_distribution_report,
    generate_obesity_analysis_report
)

router = APIRouter()

DB_FOLDER = os.path.join(os.getcwd(), "data/db")


# @router.get("/")
# def get_data_drift_report():
#     conn = sqlite3.connect(os.path.join(DB_FOLDER, "extended_data.db"))
#     query_for_original_data = """SELECT * FROM original_data"""
#     query_for_extended_data = """SELECT * FROM extended_data"""

#     df_orginal = pd.read_sql_query(query_for_original_data, conn)
#     df_extended = pd.read_sql_query(query_for_extended_data, conn)


@router.get("/data-summary")
def get_data_summary_report():
    path = generate_data_summary_report()
    if path:
        return FileResponse(path, media_type='text/html')
    return {"error": "Failed to generate report"}


@router.get("/feature-distribution")
def get_feature_distribution_report():
    path = generate_feature_distribution_report()
    if path:
        return FileResponse(path, media_type='text/html')
    return {"error": "Failed to generate report"}


@router.get("/obesity-analysis")
def get_obesity_analysis_report():
    path = generate_obesity_analysis_report()
    if path:
        return FileResponse(path, media_type='text/html')
    return {"error": "Failed to generate report"}