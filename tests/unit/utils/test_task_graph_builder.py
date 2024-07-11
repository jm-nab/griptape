import pytest
from griptape.tasks import *
from griptape.structures import *
from griptape.tools import DateTime
from tests.mocks.mock_structure_config import MockStructureConfig


class TestBuilder:
    ### Building workflows with the Workflow class directly
    def test_workflow_builder(self):
        ## Add tasks one by one, declaring parents and children
        workflow1 = (
            Workflow(config=MockStructureConfig())
            .add_task(PromptTask("test1", id="task1"))
            .add_task(PromptTask("test2", id="task2"), parents={"task1"})
            .add_task(PromptTask("test3", id="task3"), parents={"task1"})
            .add_task(PromptTask("test4", id="task4"), parents={"task2", "task3"})
        )

        ## Add parents and children to the tasks sequentially
        ## in the order they are added
        workflow3 = (
            Workflow(config=MockStructureConfig())
            ## add task1
            .add_task(PromptTask("test1", id="task1"))
            ## add children to task1
            .add_children(PromptTask("test2", id="task2"), PromptTask("test3", id="task3"))
            ## add child to tasks 2 and 3
            .add_child(PromptTask("test4", id="task4"))
        )

        ## Add tasks on init, then add parents and children by id
        workflow4 = (
            Workflow(
                config=MockStructureConfig(),
                tasks={
                    PromptTask("test1", id="task1"),
                    PromptTask("test2", id="task2"),
                    PromptTask("test3", id="task3"),
                    PromptTask("test4", id="task4"),
                },
            )
            .add_task("task1")
            .add_children("task2", "task3")
            .add_child("task4")
        )

        ## Add tasks with the add_tasks method
        ## declaring parents that apply to each task passed
        workflow6 = (
            Workflow(config=MockStructureConfig())
            .add_task(PromptTask("test1", id="task1"))
            .add_tasks(PromptTask("test2", id="task2"), PromptTask("test3", id="task3"), parents={"task1"})
            .add_task(PromptTask("test4", id="task4"), parents={"task2", "task3"})
        )

        task1 = PromptTask("test1", id="task1")
        task2 = PromptTask("test2", id="task2")
        task3 = PromptTask("test3", id="task3")
        task4 = PromptTask("test4", id="task4")

        workflow7 = Workflow(
            config=MockStructureConfig(),
            task_graph={task1: set(), task2: {task1}, task3: {task1}, task4: {task2, task3}},
        )

        workflow8 = Workflow(
            config=MockStructureConfig(),
            tasks={task1, task2, task3, task4},
            task_id_graph={"task1": set(), "task2": {"task1"}, "task3": {"task1"}, "task4": {"task2", "task3"}},
        )

        assert workflow1.task_id_graph == {
            "task1": set(),
            "task2": {"task1"},
            "task3": {"task1"},
            "task4": {"task2", "task3"},
        }
        assert workflow1.task_id_graph == workflow3.task_id_graph
        assert workflow1.task_id_graph == workflow4.task_id_graph
        assert workflow1.task_id_graph == workflow6.task_id_graph
        assert workflow1.task_id_graph == workflow7.task_id_graph
        assert workflow1.task_id_graph == workflow8.task_id_graph

        workflow1.run()
        workflow3.run()
        workflow4.run()
        workflow6.run()
        workflow7.run()
        workflow8.run()

    ## Building workflows with a pipeline
    def test_workflow_builder_with_pipeline(self):
        pipeline = Pipeline(
            config=MockStructureConfig(),
            ordered_tasks=[
                PromptTask("test1", id="task1"),
                PromptTask("test2", id="task2"),
                PromptTask("test3", id="task3"),
                PromptTask("test4", id="task4"),
            ],
        )
        workflow = Workflow(
            config=MockStructureConfig(), tasks=pipeline.tasks, task_graph=pipeline.task_graph
        ).add_task(PromptTask("extra_task", id="task_extra"), parents={"task2"}, children={"task3"})

        pipeline.run()
        workflow.run()

        assert pipeline.to_graph() == {"task1": set(), "task2": {"task1"}, "task3": {"task2"}, "task4": {"task3"}}

        assert workflow.to_graph() == {
            "task1": set(),
            "task2": {"task1"},
            "task_extra": {"task2"},
            "task3": {"task2", "task_extra"},
            "task4": {"task3"},
        }

    ## Building workflows with an Agent
    def test_workflow_builder_with_agent(self):
        agent = Agent(config=MockStructureConfig(), tools=[DateTime()])
        workflow = Workflow(config=MockStructureConfig(), tasks=agent.tasks, task_graph=agent.task_graph)

        agent.run()
        workflow.run()

        assert agent.to_graph() == {agent.task.id: set()}
        assert workflow.to_graph() == {agent.task.id: set()}
