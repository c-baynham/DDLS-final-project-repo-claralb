import json
from flask import Flask, jsonify, request
import sys
import os
from fastmcp import FastMCP

# Add the parent directory to the path to allow imports from mcp_toolset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mcp_toolset import server as mcp_server  # Your original logic
import importlib
importlib.reload(mcp_server)

# --- Flask app ---
app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({"message": "MCP Toolset Server is running."})

# Keep all other Flask @app.route endpoints unchanged
# e.g., prepare_data, train_model, etc.

# --- MCP wrapper ---
mcp = FastMCP("MCP Toolset Server")

# Wrap your functions as MCP tools
@mcp.tool()
def prepare_data(test_size: float = 0.2, random_state: int = 42) -> str:
    return mcp_server.prepare_data(test_size=test_size, random_state=random_state)

@mcp.tool()
def train_model(model_name: str) -> str:
    return mcp_server.train_model(model_name=model_name)

@mcp.tool()
def get_cv_accuracies() -> dict:
    return mcp_server.get_cv_accuracies()

@mcp.tool()
def generate_roc_curves() -> str:
    return mcp_server.generate_all_roc_curves_plot()

@mcp.tool()
def generate_feature_importance() -> str:
    return mcp_server.generate_feature_importance_subplots()

@mcp.tool()
def run_data_exploration() -> str:
    return mcp_server.run_data_exploration()

@mcp.tool()
def generate_confusion_matrices() -> str:
    return mcp_server.generate_confusion_matrices()

# --- Run MCP server ---
if __name__ == '__main__':
    print("Starting MCP server on http://127.0.0.1:7171 (HTTP transport)")
    mcp.run(transport="http", host="127.0.0.1", port=7171)
