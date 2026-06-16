from langgraph.graph import StateGraph, END
from core.state import DiagnosticState
from agents.gateway import gateway_node
from agents.cv_wrappers import create_cv_agent
from agents.consensus import consensus_node
from agents.settlement import settlement_node
from agents.reporter import reporter_node
from vision.analyzers import MODULE_ROUTING_MAP

workflow = StateGraph(DiagnosticState)

# 1. Add Nodes
workflow.add_node("gateway", gateway_node)

cv_agent_names = []
for metric_key, engine_function in MODULE_ROUTING_MAP.items():
    agent_name = f"{metric_key}_agent"
    workflow.add_node(agent_name, create_cv_agent(metric_key, engine_function))
    cv_agent_names.append(agent_name)

workflow.add_node("dermatologist_consensus", consensus_node)
workflow.add_node("settlement", settlement_node)
workflow.add_node("reporter", reporter_node)

# 2. Add Edges & Routing
workflow.set_entry_point("gateway")

def route_to_specialists(state: DiagnosticState) -> list:
    if state.get("gateway_error"):
        print(f"Halt Condition Met: {state['gateway_error']}. Direct routing to END.")
        return [END]
    return cv_agent_names

routing_map = {name: name for name in cv_agent_names}
routing_map[END] = END

workflow.add_conditional_edges("gateway", route_to_specialists, routing_map)

for agent_name in cv_agent_names:
    workflow.add_edge(agent_name, "dermatologist_consensus")

workflow.add_edge("dermatologist_consensus", "settlement")
workflow.add_edge("settlement", "reporter")
workflow.add_edge("reporter", END)

# Export the compiled app
app = workflow.compile()