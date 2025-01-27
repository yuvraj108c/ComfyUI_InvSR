from .node import LoadInvSRModels, InvSRSampler
 
NODE_CLASS_MAPPINGS = { 
    "LoadInvSRModels" : LoadInvSRModels,
    "InvSRSampler" : InvSRSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
     "LoadInvSRModels" : "Load InvSR Models",
     "InvSRSampler" : "InvSRSampler"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']