import copy
from abc import ABC, abstractmethod

import pooltool.ani.tasks as tasks
from pooltool.ani.action import Action
from pooltool.ani.globals import Global, require_showbase
from pooltool.utils.strenum import StrEnum, auto


class Mode(StrEnum):
    aim = auto()
    ball_in_hand = auto()
    calculate = auto()
    call_shot = auto()
    cam_load = auto()
    cam_save = auto()
    game_over = auto()
    menu = auto()
    pick_ball = auto()
    purgatory = auto()
    shot = auto()
    stroke = auto()
    view = auto()
    none = auto()


class BaseMode(ABC):
    keymap = None
    name = None

    def __init__(self):
        if self.keymap is None:
            raise NotImplementedError(
                "Subclasses of BaseMode must have 'keymap' attribute"
            )

        if self.name is None:
            raise NotImplementedError(
                "Subclasses of BaseMode must have 'name' attribute"
            )

        self.defaults = copy.deepcopy(self.keymap)

        tasks.add(self.shared_task, "shared_task")
        tasks.add(self.cam_save_watch, "cam_save_watch")
        tasks.add(self.cam_load_watch, "cam_load_watch")
        tasks.add(self.help_watch, "help_watch")

    def shared_task(self, task):
        if self.keymap.get(Action.quit):
            Global.base.messenger.send("close-scene")

            self.keymap[Action.quit] = False
            Global.mode_mgr.change_mode(Mode.menu)

        elif self.keymap.get(Action.introspect):
            self.keymap[Action.introspect] = False

            import pdb

            import pooltool as pt

            shot = Global.shots.active
            pdb.set_trace()

        return task.cont

    def cam_save_watch(self, task):
        if self.keymap.get(Action.cam_save) and Global.mode_mgr.mode != Mode.cam_save:
            Global.mode_mgr.change_mode(Mode.cam_save)

        return task.cont

    def cam_load_watch(self, task):
        if self.keymap.get(Action.cam_load) and Global.mode_mgr.mode != Mode.cam_load:
            Global.mode_mgr.change_mode(Mode.cam_load)

        return task.cont

    def help_watch(self, task):
        if self.keymap.get(Action.show_help):
            self.keymap[Action.show_help] = False
            if self.help_node.is_hidden():
                self.help_node.show()
            else:
                self.help_node.hide()

        return task.cont

    def update_keymap(self, action_name, action_state):
        self.keymap[action_name] = action_state

    def register_keymap_event(self, keystroke, action_name, action_state):
        """Register an event that updates the mode's keymap"""
        tasks.register_event(keystroke, self.update_keymap, [action_name, action_state])

    def reset_action_states(self):
        self.keymap = copy.deepcopy(self.defaults)

    @abstractmethod
    def enter(self):
        pass

    @abstractmethod
    def exit(self):
        pass


class ModeManager:
    def __init__(self, mode_classes):
        self.mode_classes = mode_classes

        self.baseline_events = []

        self.modes = None
        self.last_mode = None
        self.mode_stroked_from = None
        self.mode = None

    @require_showbase
    def init_modes(self):
        """Initialize the modes"""
        self.modes = {name: mode() for name, mode in self.mode_classes.items()}

    def update_event_baseline(self):
        """Update events that are listened to independent of mode

        If called, the current events being listened to (accessed with
        Global.base.messenger.get_events()) will become the new baseline. Events in the
        baseline persist when modes are changed. All other events are destroyed when the
        mode is changed.
        """
        self.baseline_events = Global.base.messenger.get_events()

    def change_mode(self, mode, exit_kwargs={}, enter_kwargs={}):
        assert mode in Mode

        # Teardown operations for the old mode
        self.end_mode(**exit_kwargs)

        # Build up operations for the new mode
        self.start_mode(mode, **enter_kwargs)

    def end_mode(self, **kwargs):
        self.last_mode = self.mode

        # Stop watching events related to mode
        self.remove_mode_events()

        if self.mode is not None:
            self.modes[self.mode].exit(**kwargs)
            self.modes[self.mode].reset_action_states()

        self.mode = None

    def start_mode(self, mode, **kwargs):
        self.mode = mode
        self.modes[mode].enter(**kwargs)

    def get_keymap(self):
        assert self.mode
        return self.modes[self.mode].keymap

    def remove_mode_events(self):
        """Stop watching for events related to the current mode"""
        for event in Global.base.messenger.get_events():
            if event in self.baseline_events:
                # This event was being watched before the mode was entered, so we leave
                # it untouched
                continue

            Global.base.ignore(event)
