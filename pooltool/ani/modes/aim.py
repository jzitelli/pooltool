#! /usr/bin/env python

import pooltool.ani as ani
import pooltool.ani.utils as autils

from pooltool.ani.modes import Mode, action

import numpy as np


class CueAvoid(object):
    def __init__(self):
        self.min_theta = 0

        # length of cue butt from cueing ball center
        self.cueing_ball_pos = self.cue.get_node('cue_stick_focus').getPos()


    def collision_task(self, task):
        max_min_theta = 0
        for entry in self.collision_handler.entries:
            min_theta = self.process_collision(entry)
            if min_theta > max_min_theta:
                max_min_theta = min_theta

        self.min_theta = max_min_theta
        return task.cont


    def process_collision(self, entry):
        cushion = self.get_cushion(entry)
        cushion_height = cushion.p1[2]
        contact_point = entry.getSurfacePoint(render.find('scene'))
        desired_contact_point = contact_point
        desired_contact_point[2] = cushion_height

        bx, by, bz = self.cueing_ball_pos
        cx, cy, cz = desired_contact_point

        min_theta = max(0, np.arctan2(cz-bz, np.sqrt((bx-cx)**2 + (by-cy)**2)) * 180/np.pi)

        return min_theta


    def get_cushion(self, entry):
        expected_suffix = 'cushion_cplane_'
        into_node_path_name = entry.get_into_node_path().name
        assert into_node_path_name.startswith(expected_suffix)
        cushion_id = into_node_path_name[len(expected_suffix):]
        return self.table.cushion_segments['linear'][cushion_id]


class AimMode(Mode, CueAvoid):
    keymap = {
        action.fine_control: False,
        action.adjust_head: False,
        action.quit: False,
        action.stroke: False,
        action.view: False,
        action.zoom: False,
        action.elevation: False,
        action.english: False,
        action.cam_save: False,
        action.cam_load: False,
        action.pick_ball: False,
        action.call_shot: False,
        action.ball_in_hand: False,
    }

    def enter(self, load_prev_cam=False):
        self.mouse.hide()
        self.mouse.relative()
        self.mouse.track()

        self.magnet_theta = True

        if not self.cue.has_focus:
            self.cue.init_focus(self.cueing_ball)
        else:
            self.cue.update_focus()

        self.cue.show_nodes()
        self.cue.get_node('cue_stick').setX(0)
        self.player_cam.update_focus(self.cueing_ball.get_node('ball').getPos())
        if load_prev_cam:
            self.player_cam.load_state('aim')

        self.task_action('escape', action.quit, True)
        self.task_action('f', action.fine_control, True)
        self.task_action('f-up', action.fine_control, False)
        self.task_action('t', action.adjust_head, True)
        self.task_action('t-up', action.adjust_head, False)
        self.task_action('mouse1', action.zoom, True)
        self.task_action('mouse1-up', action.zoom, False)
        self.task_action('s', action.stroke, True)
        self.task_action('v', action.view, True)
        self.task_action('1', action.cam_save, True)
        self.task_action('2', action.cam_load, True)
        self.task_action('q', action.pick_ball, True)
        self.task_action('c', action.call_shot, True)
        self.task_action('g', action.ball_in_hand, True)
        self.task_action('b', action.elevation, True)
        self.task_action('b-up', action.elevation, False)
        self.task_action('e', action.english, True)
        self.task_action('e-up', action.english, False)

        CueAvoid.__init__(self)

        self.add_task(self.collision_task, 'collision_task')
        self.add_task(self.aim_task, 'aim_task')


    def exit(self):
        self.remove_task('aim_task')
        self.cue.hide_nodes()
        self.player_cam.store_state('aim', overwrite=True)


    def aim_task(self, task):
        if self.keymap[action.view]:
            self.change_mode('view', enter_kwargs=dict(move_active=True))
        elif self.keymap[action.stroke]:
            self.change_mode('stroke')
        elif self.keymap[action.pick_ball]:
            self.change_mode('pick_ball')
        elif self.keymap[action.call_shot]:
            self.change_mode('call_shot')
        elif self.keymap[action.ball_in_hand]:
            self.change_mode('ball_in_hand')
        elif self.keymap[action.zoom]:
            self.zoom_camera_aim()
        elif self.keymap[action.adjust_head]:
            self.adjust_head_aim()
        elif self.keymap[action.elevation]:
            self.elevate_cue()
        elif self.keymap[action.english]:
            self.apply_english()
        else:
            self.rotate_camera_aim()

        return task.cont


    def zoom_camera_aim(self):
        with self.mouse:
            s = -self.mouse.get_dy()*ani.zoom_sensitivity

        self.player_cam.node.setPos(autils.multiply_cw(self.player_cam.node.getPos(), 1-s))


    def adjust_head_aim(self):
        with self.mouse:
            alpha_y = max(min(0, self.player_cam.focus.getR() + ani.rotate_sensitivity_y * self.mouse.get_dy()), -90)

        self.player_cam.focus.setR(alpha_y) # Move view vertically


    def rotate_camera_aim(self):
        if self.keymap[action.fine_control]:
            fx, fy = ani.rotate_fine_sensitivity_x, ani.rotate_fine_sensitivity_y
        else:
            fx, fy = ani.rotate_sensitivity_x, ani.rotate_sensitivity_y

        with self.mouse:
            alpha_x = self.player_cam.focus.getH() - fx * self.mouse.get_dx()
            alpha_y = max(min(0, self.player_cam.focus.getR() + fy * self.mouse.get_dy()), -90)

        self.player_cam.focus.setH(alpha_x) # Move view laterally
        self.player_cam.focus.setR(alpha_y) # Move view vertically

        self.fix_cue_stick_to_camera()

        current_theta = -self.cue.get_node('cue_stick_focus').getR()
        if (current_theta < self.min_theta) or self.magnet_theta:
            self.cue.get_node('cue_stick_focus').setR(-self.min_theta)
            self.hud_elements['jack'].set(self.min_theta)


    def fix_cue_stick_to_camera(self):
        self.cue.get_node('cue_stick_focus').setH(self.player_cam.focus.getH())


    def elevate_cue(self):
        cue = self.cue.get_node('cue_stick_focus')

        with self.mouse:
            delta_elevation = self.mouse.get_dy()*ani.elevate_sensitivity

        old_elevation = -cue.getR()
        new_elevation = max(0, min(ani.max_elevate, old_elevation + delta_elevation))

        if self.min_theta > new_elevation:
            # user set theta to minimum value, resume cushion tracking
            self.magnet_theta = True
            new_elevation = self.min_theta
        else:
            # theta has been modified by the user, so no longer tracks the cushion
            self.magnet_theta = False

        cue.setR(-new_elevation)

        # update hud
        self.hud_elements['jack'].set(new_elevation)



    def apply_english(self):
        with self.mouse:
            dx, dy = self.mouse.get_dx(), self.mouse.get_dy()

        cue = self.cue.get_node('cue_stick')
        R = self.cue.follow.R

        delta_y, delta_z = dx*ani.english_sensitivity, dy*ani.english_sensitivity

        # y corresponds to side spin, z to top/bottom spin
        new_y = cue.getY() + delta_y
        new_z = cue.getZ() + delta_z

        norm = np.sqrt(new_y**2 + new_z**2)
        if norm > ani.max_english*R:
            new_y *= ani.max_english*R/norm
            new_z *= ani.max_english*R/norm

        cue.setY(new_y)
        cue.setZ(new_z)

        # update hud
        a, b = -new_y/R, new_z/R
        self.hud_elements['english'].set(a, b)

