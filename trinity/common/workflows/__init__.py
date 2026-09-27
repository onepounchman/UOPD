"""Workflow module"""
from trinity.common.workflows.workflow import Task, Workflow
from trinity.utils.registry import Registry

WORKFLOWS: Registry = Registry(
    "workflows",
    default_mapping={
        "FutureBridgeAlfworldWorkflow": "trinity.common.workflows.envs.UOPD.alfworld.futurebridge_workflow.FutureBridgeAlfworldWorkflow",
        "FutureBridgeWebShopWorkflow": "trinity.common.workflows.envs.UOPD.webshop.futurebridge_workflow.FutureBridgeWebShopWorkflow",
        "OPD_alfworld_workflow": "trinity.common.workflows.envs.UOPD.alfworld.OPD_workflow.OnPolicyDistillVerlAgentAlfworldWorkflow",
        "UOPD_alfworld_workflow": "trinity.common.workflows.envs.UOPD.alfworld.UOPD_workflow.UOPDAlfworldWorkflow",
        "eval_alfworld_workflow": "trinity.common.workflows.envs.UOPD.alfworld.eval_workflow.EvalAlfworldWorkflow",
        "TCOD_f2b_alfworld_workflow": "trinity.common.workflows.envs.UOPD.alfworld.TCOD_f2b_workflow.TCOD_f2b_alfworld_workflow",
        "TCOD_b2f_alfworld_workflow": "trinity.common.workflows.envs.UOPD.alfworld.TCOD_b2f_workflow.TCOD_b2f_alfworld_workflow",
        "OPD_webshop_workflow": "trinity.common.workflows.envs.UOPD.webshop.OPD_workflow.OnPolicyDistillVerlAgentWebshopWorkflow",
        "UOPD_webshop_workflow": "trinity.common.workflows.envs.UOPD.webshop.UOPD_workflow.UOPDWebshopWorkflow",
        "eval_webshop_workflow": "trinity.common.workflows.envs.UOPD.webshop.eval_workflow.EvalWebshopWorkflow",
        "TCOD_f2b_webshop_workflow": "trinity.common.workflows.envs.UOPD.webshop.TCOD_f2b_workflow.TCOD_f2b_webshop_workflow",
        "TCOD_b2f_webshop_workflow": "trinity.common.workflows.envs.UOPD.webshop.TCOD_b2f_workflow.TCOD_b2f_webshop_workflow",
    },
)

__all__ = [
    "Task",
    "Workflow",
    "WORKFLOWS",
]
