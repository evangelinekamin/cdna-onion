"""The arena: competitors' boxes submit entries, a runner on the organizer's
machine judges them one at a time on the GPU cluster, and results go back.

model:  validation of everything a box sends, quotas, turn-taking, and the
        shape of what a box gets back
judge:  the summary step that runs on the cluster after build and rig
runner: the restart-safe event loop and its SSH links
"""
