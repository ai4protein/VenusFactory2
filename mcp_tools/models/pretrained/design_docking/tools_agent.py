"""Native tools for the shared design and docking operations."""
from langchain_core.tools import StructuredTool
from .operations import ligandmpnn_design, antibmpnn_design, vina_dock, bindcraft_design, boltzgen_design, diffdock_dock

EXTERNAL_TOOLS = [
    StructuredTool.from_function(func=ligandmpnn_design, name='ligandmpnn_design', metadata={'tool_category': 'model_prediction', 'queue_resource': 'gpu', 'timeout_seconds': 3600}),
    StructuredTool.from_function(func=antibmpnn_design, name='antibmpnn_design', metadata={'tool_category': 'model_prediction', 'queue_resource': 'gpu', 'timeout_seconds': 3600}),
    StructuredTool.from_function(func=vina_dock, name='vina_dock', metadata={'tool_category': 'molecular_docking', 'queue_resource': 'cpu', 'timeout_seconds': 3600}),
    StructuredTool.from_function(func=bindcraft_design, name='bindcraft_design', metadata={'tool_category': 'model_prediction', 'queue_resource': 'gpu', 'timeout_seconds': 3600}),
    StructuredTool.from_function(func=boltzgen_design, name='boltzgen_design', metadata={'tool_category': 'model_prediction', 'queue_resource': 'gpu', 'timeout_seconds': 3600}),
    StructuredTool.from_function(func=diffdock_dock, name='diffdock_dock', metadata={'tool_category': 'molecular_docking', 'queue_resource': 'gpu', 'timeout_seconds': 3600}),
]
