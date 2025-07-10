# %%
from rubric.examples import create_code_review_rubric

# %%
# generator = RubricTreeGenerator()
# tree = generator.generate_rubric_tree(
#     task_description="Add a new slide about kangaroos",
#     # "We are testing a computer-use agent to see if it can solve PowerPoint tasks.
#     # The current task is to add a new slide about kangaroos.",
#     max_depth=-1
# )

tree = create_code_review_rubric()

# %%
tree.print_tree()
# %%
tree.plot()
