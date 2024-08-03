#!/usr/bin/env python3

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agentic_webapp.utils import ROOT_DIR


db_path = f"{ROOT_DIR}/data/langgraph.sqlite"
async_sqlite_saver = AsyncSqliteSaver.from_conn_string(db_path)
