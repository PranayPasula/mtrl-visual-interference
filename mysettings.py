# -for global variables used in mtl visual interference experiment. 
# -kept in a separate file to avoid import conflicts.

# init function so files don't use replay_buffers before it is ready
def init():
    global replay_buffers
    replay_buffers = []