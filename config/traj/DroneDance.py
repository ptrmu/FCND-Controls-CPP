import numpy as np
import Trajectory4Poly2Min
import warnings
# import the quaternion module but suppress the warning about numba
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import quaternion


class DroneDance:
    def __init__(self, num_drones):
        self.waypoint_interval = 0.5
        self.trajectory_interval = 0.02
        self.current_t = 0.0
        self.ts = []
        self.num_drones = num_drones
        self.drones = [Drone(i) for i in range(num_drones)]
        self.generate_waypoint(0.0)

    def push_frame(self, factory, drones=None):
        if drones is None:
            drones = self.drones
        factory.prep_frame(drones)
        for drone in drones:
            drone.push_frame(factory)

    def pop_frame(self, drones=None):
        if drones is None:
            drones = self.drones
        for drone in drones:
            drone.pop_frame()

    def generate_waypoint(self, interval):
        self.current_t += interval
        self.ts.append(self.current_t)
        for drone in self.drones:
            drone.generate_waypoint(interval)

    def generate_waypoints(self, duration, interval=None):
        waypoint_interval = self.waypoint_interval if interval is None else interval
        waypoint_interval = min(duration, waypoint_interval)
        current = waypoint_interval
        while current < duration + 0.00001:
            current += waypoint_interval
            self.generate_waypoint(waypoint_interval)

    def save_waypoints(self, file_name):
        for drone in self.drones:
            with open(file_name + "{0:02d}".format(drone.idx) + ".txt", 'w') as the_file:
                t0 = 0
                p0 = np.zeros(3)
                v0 = np.zeros(3)
                v = np.zeros(3)
                a = np.zeros(3)
                for t, p in zip(self.ts, drone.waypoints):
                    if t > 0.0:
                        td = t - t0
                        v = (p - p0) / td
                        a = (v - v0) / td
                    the_file.write("{0:f},{1:f},{2:f},{3:f},".format(t, p[0], p[1], p[2]))
                    the_file.write("{0:f},{1:f},{2:f},".format(v[0], v[1], v[2]))
                    the_file.write("{0:f},{1:f},{2:f}\n".format(a[0], a[1], a[2]))
                    t0 = t
                    p0 = p
                    v0 = v

    def save_trajectories(self, file_name):
        path3ds = [(drone.idx, Path3d(self.ts, drone)) for drone in self.drones]
        for idx, path3d in path3ds:
            with open(file_name + "{0:02d}".format(idx) + ".txt", 'w') as the_file:
                for t in np.arange(0.0, self.current_t, self.trajectory_interval):
                    the_file.write(path3d.at(t))


class Path3d:
    def __init__(self, ts, drone):
        waypoints = np.array(drone.waypoints)
        self.x = Trajectory4Poly2Min.Trajectory4Poly2Min(waypoints[:, 0], ts, 0.0, 0.0)
        self.y = Trajectory4Poly2Min.Trajectory4Poly2Min(waypoints[:, 1], ts, 0.0, 0.0)
        self.z = Trajectory4Poly2Min.Trajectory4Poly2Min(waypoints[:, 2], ts, 0.0, 0.0)

    def at(self, t):
        s = "{0:f},{1:f},{2:f},{3:f},{4:f},{5:f},{6:f},{7:f},{8:f},{9:f}\n".format(
                t,
                self.x.calc_y(t), self.y.calc_y(t), self.z.calc_y(t),
                self.x.calc_y_dot(t), self.y.calc_y_dot(t), self.z.calc_y_dot(t),
                self.x.calc_y_dotdot(t), self.y.calc_y_dotdot(t), self.z.calc_y_dotdot(t))
        return s


class Drone:
    def __init__(self, idx):
        self.idx = idx
        self.current_local_position = np.zeros(3)
        self.current_global_position = np.zeros(3)
        self.frame_list = FrameList()
        self.waypoints = []

    def push_frame(self, factory):
        frame = factory.new(self)
        self.frame_list.push_frame(frame)
        self.current_local_position = self.frame_list.global_2_local(self.current_global_position)

    def pop_frame(self):
        self.frame_list.pop_frame()
        self.current_local_position = self.frame_list.global_2_local(self.current_global_position)

    def generate_waypoint(self, interval):
        self.frame_list.advance(interval)
        position = self.frame_list.local_2_global(self.current_local_position)
        self.current_global_position = position
        self.waypoints.append(position)


class Transform:
    def __init__(self, rotate=None, translate=None, transform=None):
        if transform is not None:
            rotate = transform.rotate if rotate is None else rotate
            translate = transform.translate if translate is None else translate
        self.rotate = quaternion.one if rotate is None else np.copy(rotate)
        self.translate = np.zeros(3) if translate is None else np.copy(translate)


class Frame:
    def __init__(self, factory, transform=None):
        self.factory = factory
        self.transform = Transform(transform=transform)

    def local_2_global(self, local_position):
        return quaternion.rotate_vectors(self.transform.rotate.conj(), local_position) + self.transform.translate

    def global_2_local(self, global_position):
        return quaternion.rotate_vectors(self.transform.rotate, global_position - self.transform.translate)

    def advance(self, interval):
        pass


class FrameList:
    def __init__(self):
        self.frames = []

    def local_2_global(self, local_position):
        position = local_position
        for frame in reversed(self.frames):
            position = frame.local_2_global(position)
        return position

    def global_2_local(self, global_position):
        position = global_position
        for frame in self.frames:
            position = frame.global_2_local(position)
        return position

    def advance(self, interval):
        for frame in self.frames:
            frame.advance(interval)

    def push_frame(self, frame):
        self.frames.append(frame)

    def pop_frame(self):
        self.frames.pop()


class FrameFactory:
    def __init__(self, drone_dance):
        self.drone_dance = drone_dance

    def new(self, drone):
        return self._Frame(self, drone)

    def prep_frame(self, drones):
        pass


class CombinedFrameFactory(FrameFactory):
    def __init__(self, drone_dance, factory_list):
        super().__init__(drone_dance)
        self.factory_list = factory_list

    class _Frame(Frame):
        def __init__(self, factory, drone):
            super().__init__(factory)
            self.frame_list = FrameList()
            for factory in factory.factory_list:
                frame = factory.new(drone)
                self.frame_list.push_frame(frame)

        def local_2_global(self, local_position):
            return self.frame_list.local_2_global(local_position)

        def global_2_local(self, global_position):
            return self.frame_list.global_2_local(global_position)

        def advance(self, interval):
            self.frame_list.advance(interval)


class UniformAngularDistributionFrameFactory(FrameFactory):
    def __init__(self, drone_dance, rotation_axis):
        super().__init__(drone_dance)
        self.rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    class _Frame(Frame):
        def __init__(self, factory, drone):
            separation_angle = 2.0 * np.pi / factory.drone_dance.num_drones
            rotation_norm = float(drone.idx) * separation_angle
            rotate = quaternion.from_rotation_vector(factory.rotation_axis * rotation_norm)
            super().__init__(factory, Transform(rotate=rotate))


class TransformRate:
    def __init__(self, rotation_axis=None, rotation_rate=None,
                 translate_rate=None, transform_rate=None):
        if transform_rate is not None:
            rotation_axis = translate_rate.rotation_axis if rotation_axis is None else rotation_axis
            rotation_rate = translate_rate.rotation_rate if rotation_rate is None else rotation_rate
            translate_rate = transform_rate.translate_rate if translate_rate is None else translate_rate
        self.rotation_axis = np.array([0, 0, 1.0] if rotation_axis is None else rotation_axis)
        self.rotation_rate = float(0 if rotation_rate is None else rotation_rate)
        self.translate_rate = np.zeros(3) if translate_rate is None else translate_rate

    def set_x_rate(self, x_rate):
        self.translate_rate[0] = x_rate

    def set_y_rate(self, y_rate):
        self.translate_rate[1] = y_rate

    def set_z_rate(self, z_rate):
        self.translate_rate[2] = z_rate

    def set_rotation_rate(self, rotation_rate):
        self.rotation_rate = float(rotation_rate)

    def set_rotation_axis(self, rotation_axis):
        self.rotation_axis = np.array(rotation_axis) / np.linalg.norm(rotation_axis)


class ConstantRateFrameFactory(FrameFactory):
    def __init__(self, drone_dance, transform_rate, transform=None):
        super().__init__(drone_dance)
        self.transform = transform
        self.transform_rate = transform_rate

    class _Frame(Frame):
        def __init__(self, factory, drone):
            super().__init__(factory, factory.transform)
            self.factory = factory

        def advance(self, interval):
            r = quaternion.from_rotation_vector(
                self.factory.transform_rate.rotation_axis *
                self.factory.transform_rate.rotation_rate *
                interval)
            self.transform.rotate = r * self.transform.rotate
            self.transform.translate += self.factory.transform_rate.translate_rate * interval


class FormationFrameFactory(FrameFactory):
    """
    FormationFrameFactory base class will generate trajectories to move
    drones from their current arrangement to a new formation.
    The trajectory will be such that the drones arrive
    at their individual locations after formation_duration time.
    Making formation_duration longer or shorter will decrease or
    increase the speed of the drones while setting up the
    formation.
    """
    def __init__(self, drone_dance, formation_duration):
        super().__init__(drone_dance)
        self.formation_duration = formation_duration
        self.formation_locations = {}

    def prep_frame(self, drones):
        for drone, location in zip(drones, self.location_generator(drones)):
            self.formation_locations[drone.idx] = location

    class _Frame(Frame):
        def __init__(self, factory, drone):
            super().__init__(factory)
            formation_location = factory.formation_locations[drone.idx]
            self.velocity = (formation_location - drone.current_local_position) / self.factory.formation_duration

        def advance(self, interval):
            self.transform.translate += interval * self.velocity


class DiamondFormationFrameFactory(FormationFrameFactory):
    def __init__(self, drone_dance, formation_duration, separation_factor, z):
        super().__init__(drone_dance, formation_duration)
        self.separation_factor = separation_factor  # 0.24
        self.z = z

    def location_generator(self, drones):
        n = len(drones)
        s = int(np.ceil(np.exp(np.log(n)/2)))
        o = - (s - 1) / 2
        d = np.sqrt(3)

        r = self.separation_factor
        for i in range(n):
            tx = float(np.floor(i / s)) + o
            ty = float(np.mod(i, s)) + o
            # print([tx, ty], [(tx - ty) * d, tx + ty, self.z])
            yield np.array([r * (tx - ty) * d, r * (tx + ty), self.z])


class TetrahedronFormationFrameFactory(FormationFrameFactory):
    def __init__(self, drone_dance, formation_duration, separation_factor):
        super().__init__(drone_dance, formation_duration)
        self.separation_factor = separation_factor

    @staticmethod
    def p(p1, p2, f):
        return p1 + (p2 - p1) * f

    def location_generator(self, drones):
        n = int(np.ceil((len(drones) - 4) / 6.0))
        p1 = np.array([1, 1, 1]) * self.separation_factor
        p2 = np.array([-1, -1, 1]) * self.separation_factor
        p3 = np.array([1, -1, -1]) * self.separation_factor
        p4 = np.array([-1, 1, -1]) * self.separation_factor

        yield p1
        yield p2
        yield p3
        yield p4
        for i in range(n):
            f = 1.0 * (i + 1) / (1.0 + n)
            yield self.p(p1, p2, f)
            yield self.p(p1, p3, f)
            yield self.p(p1, p4, f)
            yield self.p(p2, p3, f)
            yield self.p(p3, p4, f)
            yield self.p(p4, p2, f)


class SmileyFormationFrameFactory(FormationFrameFactory):
    def __init__(self, drone_dance, formation_duration, separation_factor, z):
        super().__init__(drone_dance, formation_duration)
        self.separation_factor = separation_factor  # 0.24
        self.z = z

    def p(self, theta, r):
        t = np.pi * theta / 180.
        return np.array([r * np.cos(t), r * np.sin(t), self.z])

    def q(self, theta, f):
        return self.p(theta, self.separation_factor * f)

    def r(self, w, f):
        return self.q(180. + w * (f - 1. / 2.), 1.0)

    def location_generator(self, drones):
        eye_separation_angle = 60.
        smile_width_angle = 110.
        eye_fraction = 0.7
        while True:
            yield self.p(0.0, 0.0)
            yield self.q(eye_separation_angle / 2, eye_fraction)
            yield self.q(-eye_separation_angle / 2, eye_fraction)
            yield self.r(smile_width_angle, 0.)
            yield self.r(smile_width_angle, 0.9/3.)
            yield self.r(smile_width_angle, 2.1/3.)
            yield self.r(smile_width_angle, 1.)


if __name__ == "__main__":
    def rotate_in_and_out(dd):
        initial_height = 2
        initial_radius = 1.3
        initial_duration = 2.0
        uad = UniformAngularDistributionFrameFactory(dd, np.array([0, 0, 1]))
        dd.push_frame(uad)

        tr1 = TransformRate(translate_rate=np.array([initial_radius, 0, -initial_height]) / initial_duration)
        cr1 = ConstantRateFrameFactory(dd, tr1)
        dd.push_frame(cr1)
        dd.generate_waypoints(initial_duration, interval=initial_duration)
        dd.pop_frame()

        tr2 = TransformRate(rotation_axis=[0, 0, 1], rotation_rate=2*np.pi/8)
        cr2 = ConstantRateFrameFactory(dd, tr2)
        dd.push_frame(cr2)

        first_delta_radius = 1
        first_duration = 1
        tr3 = TransformRate(translate_rate=np.array([first_delta_radius, 0, 0]) / first_duration)
        cr3 = ConstantRateFrameFactory(dd, tr3)
        dd.push_frame(cr3)
        dd.generate_waypoints(first_duration)

        second_delta_radius = -1.5
        second_interval = 1.5
        tr3.set_x_rate(second_delta_radius / second_interval)
        dd.generate_waypoints(second_interval)

        third_delta_radius = 1.5
        third_interval = 1.5
        tr3.set_x_rate(third_delta_radius / third_interval)
        dd.generate_waypoints(third_interval)

        fourth_delta_radius = -3.5
        fourth_interval = 3.5
        tr3.set_x_rate(fourth_delta_radius / fourth_interval)
        dd.generate_waypoints(fourth_interval)

        fifth_delta_radius = 0
        fifth_interval = 3
        tr3.set_x_rate(fifth_delta_radius / fifth_interval)
        dd.generate_waypoints(fifth_interval)

    def rotate_up_and_down(dd):
        initial_height = 1
        initial_radius = 2
        initial_duration = 1.0
        uad = UniformAngularDistributionFrameFactory(dd, np.array([0, 0, 1]))
        dd.push_frame(uad)

        tr1 = TransformRate(translate_rate=np.array([initial_radius, 0, -initial_height]) / initial_duration)
        cr1 = ConstantRateFrameFactory(dd, tr1)
        dd.push_frame(cr1)
        dd.generate_waypoints(initial_duration, interval=initial_duration)
        dd.pop_frame()

        tr2 = TransformRate(rotation_axis=[0, 0, 1], rotation_rate=2*np.pi/8)
        cr2 = ConstantRateFrameFactory(dd, tr2)
        dd.push_frame(cr2)

        first_delta_radius = -1.5
        first_duration = 1.5
        tr3 = TransformRate(translate_rate=np.array([first_delta_radius, 0, first_delta_radius]) / first_duration)
        cr3 = ConstantRateFrameFactory(dd, tr3)
        dd.push_frame(cr3)
        dd.generate_waypoints(first_duration)

        second_delta_radius = 2
        second_duration = 2
        tr3.set_x_rate(second_delta_radius / second_duration)
        tr3.set_z_rate(second_delta_radius / second_duration)
        dd.generate_waypoints(second_duration)

        tr3.set_x_rate(-second_delta_radius / second_duration)
        tr3.set_z_rate(-second_delta_radius / second_duration)
        dd.generate_waypoints(second_duration)

        tr3.set_x_rate(second_delta_radius / second_duration)
        tr3.set_z_rate(second_delta_radius / second_duration)
        dd.generate_waypoints(second_duration)

        tr3.set_x_rate(-second_delta_radius / second_duration)
        tr3.set_z_rate(-second_delta_radius / second_duration)
        dd.generate_waypoints(second_duration)

        tr3.set_x_rate(second_delta_radius / second_duration)
        tr3.set_z_rate(second_delta_radius / second_duration)
        dd.generate_waypoints(second_duration)

    def ferris_wheel(dd):
        center_height = 1.8
        tr1 = TransformRate(rotation_axis=[0, 1, 0])
        cr1 = ConstantRateFrameFactory(dd, tr1, transform=Transform(translate=np.array([0, 0, -center_height*1.1])))
        dd.push_frame(cr1)

        tr2 = TransformRate(rotation_axis=[0, 0, 1])
        cr2 = ConstantRateFrameFactory(dd, tr2)
        dd.push_frame(cr2)

        uad = UniformAngularDistributionFrameFactory(dd, np.array([0, 0, 1]))
        dd.push_frame(uad)

        initial_height = center_height
        initial_radius = center_height
        initial_duration = center_height

        tr3 = TransformRate(translate_rate=np.array([initial_radius, 0, -initial_height]) / initial_duration)
        cr3 = ConstantRateFrameFactory(dd, tr3, transform=Transform(translate=np.array([0, 0, center_height])))
        dd.push_frame(cr3)
        dd.generate_waypoints(initial_duration, interval=initial_duration)
        dd.pop_frame()

        tr2.set_rotation_rate(2*np.pi/4)
        dd.generate_waypoints(1)

        tr1.set_rotation_rate(2*np.pi/8)
        dd.generate_waypoints(12)

    def split_ferris_wheel(dd):
        center_height = 1.8

        drones_a = dd.drones[0::2]
        drones_b = dd.drones[1::2]
        tr1a = TransformRate(rotation_axis=[0, 1, 0])
        cr1a = ConstantRateFrameFactory(dd, tr1a, transform=Transform(translate=np.array([0, 0, -center_height*1.2])))
        dd.push_frame(cr1a, drones=drones_a)
        tr1b = TransformRate(rotation_axis=[0, 1, 0])
        cr1b = ConstantRateFrameFactory(dd, tr1b, transform=Transform(translate=np.array([0, 0, -center_height*1.2])))
        dd.push_frame(cr1b, drones=drones_b)

        tr2 = TransformRate(rotation_axis=[0, 0, 1])
        cr2 = ConstantRateFrameFactory(dd, tr2)
        dd.push_frame(cr2)

        uad = UniformAngularDistributionFrameFactory(dd, np.array([0, 0, 1]))
        dd.push_frame(uad)

        initial_height = center_height
        initial_radius = center_height
        initial_duration = center_height

        tr3 = TransformRate(translate_rate=np.array([initial_radius, 0, -initial_height]) / initial_duration)
        cr3 = ConstantRateFrameFactory(dd, tr3, transform=Transform(translate=np.array([0, 0, center_height])))
        dd.push_frame(cr3)
        dd.generate_waypoints(initial_duration, interval=initial_duration)
        dd.pop_frame()

        tr2.set_rotation_rate(2*np.pi/4)
        dd.generate_waypoints(0.2)

        tr1a.set_rotation_rate(2*np.pi/8)
        tr1b.set_rotation_rate(-2*np.pi/8)
        dd.generate_waypoints(12)

    def diamond_formation(dd):
        center_height = 2.
        tr1 = TransformRate(rotation_axis=[0, 1, 0])
        cr1 = ConstantRateFrameFactory(dd, tr1, transform=Transform(translate=np.array([0, 0, -center_height])))
        dd.push_frame(cr1)

        tr2 = TransformRate(rotation_axis=[1, 0, 0])
        cr2 = ConstantRateFrameFactory(dd, tr2)
        dd.push_frame(cr2)

        diamond_height = 0.5
        diamond_duration = 1.0
        df = DiamondFormationFrameFactory(dd, diamond_duration, 0.24, center_height - diamond_height)
        dd.push_frame(df)
        dd.generate_waypoints(diamond_duration, interval=diamond_duration)
        dd.pop_frame()

        tr2.set_rotation_rate(2*np.pi/4)
        dd.generate_waypoints(1)

        tr1.set_rotation_rate(2*np.pi/10)
        dd.generate_waypoints(12)

    def diamond_formation_from_circle(dd):
        center_height = 2.0
        tr1 = TransformRate(rotation_axis=[0, 1, 0])
        cr1 = ConstantRateFrameFactory(dd, tr1, transform=Transform(translate=np.array([0, 0, -center_height])))
        dd.push_frame(cr1)

        tr2 = TransformRate(rotation_axis=[1, 0, 0])
        cr2 = ConstantRateFrameFactory(dd, tr2)
        dd.push_frame(cr2)

        circle_height = center_height / 2
        circle_radius = center_height
        circle_duration = 2.0
        uad = UniformAngularDistributionFrameFactory(dd, np.array([0, 0, 1]))
        tr3 = TransformRate(translate_rate=np.array([circle_radius, 0, -circle_height]) / circle_duration)
        cr3 = ConstantRateFrameFactory(dd, tr3, transform=Transform(translate=np.array([0, 0, center_height])))
        dd.push_frame(uad)
        dd.push_frame(cr3)
        dd.generate_waypoints(circle_duration, interval=circle_duration)
        dd.pop_frame()
        dd.pop_frame()

        diamond_height = 0.5
        diamond_duration = 2.0
        df = DiamondFormationFrameFactory(dd, diamond_duration, 0.24, center_height - diamond_height)
        dd.push_frame(df)
        dd.generate_waypoints(diamond_duration, interval=diamond_duration)
        dd.pop_frame()

        tr2.set_rotation_rate(2*np.pi/4)
        dd.generate_waypoints(1)

        tr1.set_rotation_rate(2*np.pi/10)
        dd.generate_waypoints(12)

    def diamond_formation_from_split_ferris(dd):
        center_height = 2.0

        drones_a = dd.drones[0::2]
        drones_b = dd.drones[1::2]
        tr1a = TransformRate(rotation_axis=[0, 1, 0])
        cr1a = ConstantRateFrameFactory(dd, tr1a, transform=Transform(translate=np.array([0, 0, -center_height])))
        dd.push_frame(cr1a, drones=drones_a)
        tr1b = TransformRate(rotation_axis=[0, 1, 0])
        cr1b = ConstantRateFrameFactory(dd, tr1b, transform=Transform(translate=np.array([0, 0, -center_height])))
        dd.push_frame(cr1b, drones=drones_b)

        tr2 = TransformRate(rotation_axis=[0, 0, 1])
        cr2 = ConstantRateFrameFactory(dd, tr2)
        dd.push_frame(cr2)

        initial_height = center_height
        initial_radius = center_height * 0.9
        initial_duration = 1.5
        uad = UniformAngularDistributionFrameFactory(dd, np.array([0, 0, 1]))
        tr3 = TransformRate(translate_rate=np.array([initial_radius, 0, -initial_height]) / initial_duration)
        cr3 = ConstantRateFrameFactory(dd, tr3, transform=Transform(translate=np.array([0, 0, center_height])))
        dd.push_frame(uad)
        dd.push_frame(cr3)
        dd.generate_waypoints(initial_duration, interval=initial_duration)
        dd.pop_frame()
        dd.pop_frame()

        tr2.set_rotation_rate(2*np.pi/4)
        dd.generate_waypoints(0.5)

        tr1a.set_rotation_rate(2*np.pi/8)
        tr1b.set_rotation_rate(-2*np.pi/8)
        dd.generate_waypoints(4)

        tr2.set_rotation_rate(0)
        tr1a.set_rotation_rate(0)
        tr1b.set_rotation_rate(0)

        diamond_height = 0.5
        diamond_duration = 1.0
        df = DiamondFormationFrameFactory(dd, diamond_duration, 0.24, -center_height + diamond_height)
        dd.push_frame(df)
        dd.generate_waypoints(diamond_duration, interval=diamond_duration)
        dd.pop_frame()

        tr2.set_rotation_axis([1, 0, 0])
        tr2.set_rotation_rate(-2 * np.pi / 4)

        tr1a.set_rotation_rate(-2*np.pi/8)
        tr1b.set_rotation_rate(-2*np.pi/8)
        dd.generate_waypoints(9)

    def tetrahedron_from_diamond_formation(dd):
        center_height = 2.
        tr1 = TransformRate(rotation_axis=[0, 1, 0])
        cr1 = ConstantRateFrameFactory(dd, tr1, transform=Transform(translate=np.array([0, 0, -center_height])))
        dd.push_frame(cr1)

        tr2 = TransformRate(rotation_axis=[1, 0, 0])
        cr2 = ConstantRateFrameFactory(dd, tr2)
        dd.push_frame(cr2)

        diamond_height = 0.75
        diamond_duration = 1.0
        df = DiamondFormationFrameFactory(dd, diamond_duration, 0.24, center_height - diamond_height)
        dd.push_frame(df)
        dd.generate_waypoints(diamond_duration, interval=diamond_duration)
        dd.pop_frame()

        # tr2.set_rotation_rate(2*np.pi/4)
        # dd.generate_waypoints(3)

        tr1.set_rotation_rate(2*np.pi/4)
        dd.generate_waypoints(3.0)

        tetrahedron_duration = 2.0
        tf = TetrahedronFormationFrameFactory(dd, tetrahedron_duration, 1.0)
        dd.push_frame(tf)
        dd.generate_waypoints(tetrahedron_duration, interval=tetrahedron_duration)
        dd.pop_frame()

        dd.generate_waypoints(7)

    def tetrahedron_from_circle_formation(dd):
        center_height = 2.0
        tr1 = TransformRate(rotation_axis=[0, 1, 0])
        cr1 = ConstantRateFrameFactory(dd, tr1, transform=Transform(translate=np.array([0, 0, -center_height])))
        dd.push_frame(cr1)

        tr2 = TransformRate(rotation_axis=[1, 0, 0])
        cr2 = ConstantRateFrameFactory(dd, tr2)
        dd.push_frame(cr2)

        circle_height = center_height
        circle_radius = center_height / 2
        circle_duration = 2.0
        uad = UniformAngularDistributionFrameFactory(dd, np.array([0, 0, 1]))
        tr3 = TransformRate(translate_rate=np.array([circle_radius, 0, -circle_height]) / circle_duration)
        cr3 = ConstantRateFrameFactory(dd, tr3, transform=Transform(translate=np.array([0, 0, center_height])))
        dd.push_frame(uad)
        dd.push_frame(cr3)
        dd.generate_waypoints(circle_duration, interval=circle_duration)
        dd.pop_frame()
        dd.pop_frame()

        tetrahedron_duration = 2.0
        tf = TetrahedronFormationFrameFactory(dd, tetrahedron_duration, 1.0)
        dd.push_frame(tf)
        dd.generate_waypoints(tetrahedron_duration, interval=tetrahedron_duration)
        dd.pop_frame()

        tr2.set_rotation_rate(2*np.pi/4)
        dd.generate_waypoints(1)

        tr1.set_rotation_rate(2*np.pi/10)
        dd.generate_waypoints(12)

    def tetrahedron_from_smiley_formation(dd):
        center_height = 2.
        ra1 = np.array([0., 1., 0.])
        rm1 = -np.pi/2
        rt1 = quaternion.from_rotation_vector(ra1 * rm1)
        tf1 = Transform(translate=np.array([0, 0, -center_height]), rotate=rt1)
        tr1 = TransformRate(rotation_axis=ra1)
        cr1 = ConstantRateFrameFactory(dd, tr1, transform=tf1)
        dd.push_frame(cr1)

        smiley_height = 0.75
        smiley_radius = 0.75
        smiley_duration = 1.0
        df = SmileyFormationFrameFactory(dd, smiley_duration, smiley_radius, center_height - smiley_height)
        dd.push_frame(df)
        dd.generate_waypoints(smiley_duration, interval=smiley_duration)
        dd.pop_frame()

        # tr2.set_rotation_rate(2*np.pi/4)
        # dd.generate_waypoints(3)

        dd.generate_waypoints(5.0)

        tetrahedron_duration = 2.0
        tf = TetrahedronFormationFrameFactory(dd, tetrahedron_duration, 1.0)
        dd.push_frame(tf)
        dd.generate_waypoints(tetrahedron_duration, interval=tetrahedron_duration)
        dd.pop_frame()

        tr1.set_rotation_rate(-2*np.pi/4)
        dd.generate_waypoints(7)

    def main_test(num_drones):
        dd = DroneDance(num_drones)

        # rotate_in_and_out(dd)
        # rotate_up_and_down(dd)
        # ferris_wheel(dd)
        # split_ferris_wheel(dd)
        # diamond_formation(dd)
        # diamond_formation_from_circle(dd)
        # diamond_formation_from_split_ferris(dd)
        # tetrahedron_from_diamond_formation(dd)
        # tetrahedron_from_circle_formation(dd)
        tetrahedron_from_smiley_formation(dd)

        filename = "DroneDance" + "{0:02d}".format(num_drones)
        # dd.save_waypoints(filename)
        dd.save_trajectories(filename)
        pass

    # main_test(4)
    main_test(16)
