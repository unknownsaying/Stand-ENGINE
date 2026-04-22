"""
JOJO Stand Engine - 终极扩展版
包含：动态战斗模拟（位置偏移）、高级进化动画（形状匹配）、导出材质/纹理、配置保存/加载
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Callable
from enum import Enum
import itertools
import json
import os
from scipy.spatial import ConvexHull, KDTree
from scipy.spatial.transform import Rotation as R
import copy

# -------------------- 基础数据结构（略有扩展）--------------------
class StatRank(Enum):
    E = 1; D = 2; C = 3; B = 4; A = 5; INFINITE = 6

@dataclass
class StandStats:
    power: StatRank
    speed: StatRank
    range: StatRank
    durability: StatRank
    precision: StatRank
    potential: StatRank

class PolyhedronGeometry:
    """多面体几何生成器（基本不变，但增加获取顶点数组的方法）"""
    def __init__(self, stats: StandStats):
        self.stats = stats
        self.vertices = []
        self.edges = []
        self.faces = []
        self.radius = self._calculate_radius()
        self._generate_polyhedron()
    
    def _calculate_radius(self):
        base = 1.0
        mult = {StatRank.E:0.6, StatRank.D:0.8, StatRank.C:1.0,
                StatRank.B:1.3, StatRank.A:1.6, StatRank.INFINITE:2.0}
        return base * mult[self.stats.range]
    
    def _fibonacci_sphere(self, n):
        pts = []
        phi = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            y = 1 - (i/(n-1))*2
            r = np.sqrt(1 - y*y)
            theta = phi * i
            pts.append([np.cos(theta)*r, y, np.sin(theta)*r])
        return np.array(pts)
    
    def _perturbed_sphere(self, n):
        pts = self._fibonacci_sphere(n)
        noise = np.random.randn(*pts.shape) * 0.2 * (6 - self.stats.precision.value)
        pts += noise
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    def _generate_polyhedron(self):
        n_verts = 8 + (self.stats.power.value + self.stats.durability.value) * 3
        n_verts = min(n_verts, 100)
        if self.stats.precision.value >= 4:
            sphere_pts = self._fibonacci_sphere(n_verts)
        else:
            sphere_pts = self._perturbed_sphere(n_verts)
        sphere_pts *= self.radius
        hull = ConvexHull(sphere_pts)
        self.vertices = sphere_pts[hull.vertices].tolist()
        self.faces = hull.simplices.tolist()
        edge_set = set()
        for f in self.faces:
            for i in range(len(f)):
                edge = tuple(sorted([f[i], f[(i+1)%len(f)]]))
                edge_set.add(edge)
        self.edges = list(edge_set)
        self._add_speed_edges()
        if self.stats.potential.value >= 4:
            self._subdivide_faces()
    
    def _add_speed_edges(self):
        if self.stats.speed.value <= 2: return
        verts = np.array(self.vertices)
        existing = set(self.edges)
        ratio = {3:0.1, 4:0.2, 5:0.3, 6:0.5}[self.stats.speed.value]
        n_extra = int(len(self.edges) * ratio)
        dists = []
        for i, j in itertools.combinations(range(len(verts)), 2):
            if (i,j) not in existing and (j,i) not in existing:
                d = np.linalg.norm(verts[i]-verts[j])
                if d < self.radius*2.0:
                    dists.append((d,i,j))
        dists.sort(key=lambda x:x[0])
        for _, i, j in dists[:n_extra]:
            self.edges.append((i,j))
    
    def _subdivide_faces(self):
        level = self.stats.potential.value - 3
        for _ in range(level):
            new_verts = self.vertices.copy()
            new_faces = []
            new_edges = set(self.edges)
            vmap = {}
            def mid(a,b):
                key = tuple(sorted([a,b]))
                if key not in vmap:
                    m = (np.array(self.vertices[a])+np.array(self.vertices[b]))/2
                    m = m / np.linalg.norm(m) * self.radius
                    vmap[key] = len(new_verts)
                    new_verts.append(m.tolist())
                return vmap[key]
            for f in self.faces:
                if len(f)==3:
                    v0,v1,v2 = f
                    m01 = mid(v0,v1); m12 = mid(v1,v2); m20 = mid(v2,v0)
                    for e in [(m01,m12),(m12,m20),(m20,m01),(v0,m01),(v1,m01),
                              (v1,m12),(v2,m12),(v2,m20),(v0,m20)]:
                        new_edges.add(tuple(sorted(e)))
                    new_faces.extend([[v0,m01,m20],[v1,m12,m01],[v2,m20,m12],[m01,m12,m20]])
                else:
                    new_faces.append(f)
            self.vertices = new_verts
            self.faces = new_faces
            self.edges = list(new_edges)
    
    def get_statistics(self):
        vol = 0.0
        verts = np.array(self.vertices)
        for f in self.faces:
            if len(f)>=3:
                v0 = verts[f[0]]
                for i in range(1,len(f)-1):
                    v1 = verts[f[i]]
                    v2 = verts[f[i+1]]
                    vol += np.dot(v0, np.cross(v1,v2))/6.0
        return {
            'V': len(self.vertices), 'E': len(self.edges), 'F': len(self.faces),
            'Euler_Characteristic': len(self.vertices)-len(self.edges)+len(self.faces),
            'Radius': self.radius, 'Volume': abs(vol)
        }

class Stand:
    def __init__(self, name, master, stats, ability=""):
        self.name = name
        self.master = master
        self.stats = stats
        self.ability = ability
        self.geometry = PolyhedronGeometry(stats)
        # 世界坐标位置（用于战斗模拟）
        self.position = np.array([0.0, 0.0, 0.0])
        self.color = self._get_color()
    
    def _get_color(self):
        max_stat = max(self.stats.__dict__.items(), key=lambda x:x[1].value)[0]
        return {'power':(0.8,0.2,0.2), 'speed':(0.2,0.8,0.2),
                'range':(0.2,0.4,0.9), 'durability':(0.8,0.8,0.2),
                'precision':(0.9,0.9,0.9), 'potential':(0.7,0.3,0.7)}[max_stat]
    
    def get_transformed_vertices(self) -> np.ndarray:
        """返回经过位置平移后的顶点坐标"""
        return np.array(self.geometry.vertices) + self.position
    
    def to_dict(self) -> dict:
        """序列化为字典，用于保存配置"""
        return {
            'name': self.name,
            'master': self.master,
            'ability': self.ability,
            'stats': {k: v.value for k, v in self.stats.__dict__.items()}
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        stats = StandStats(**{k: StatRank(v) for k, v in data['stats'].items()})
        return cls(data['name'], data['master'], stats, data['ability'])

# -------------------- 1. 动态战斗模拟（位置偏移）--------------------
class DynamicCombatSimulator:
    """
    动态战斗模拟：替身可在空间中移动，实时计算交集体积
    """
    def __init__(self, stand1: Stand, stand2: Stand, initial_distance: float = 3.0):
        self.s1 = stand1
        self.s2 = stand2
        # 设置初始位置：沿X轴分布，距离受双方射程影响
        self.update_positions_by_range(initial_distance)
        self.fig = None
        self.ax = None
        self.ani = None
        self.slider_dist = None
        self.slider_angle = None
    
    def update_positions_by_range(self, separation: float):
        """根据射程属性影响相对位置：射程短的替身会更靠近中心"""
        r1 = self.s1.geometry.radius
        r2 = self.s2.geometry.radius
        # 加权位置：射程大的一方占据更多空间
        total_range_factor = r1 + r2
        if total_range_factor > 0:
            offset1 = separation * (r2 / total_range_factor)
            offset2 = separation * (r1 / total_range_factor)
        else:
            offset1 = offset2 = separation/2
        self.s1.position = np.array([-offset1, 0.0, 0.0])
        self.s2.position = np.array([offset2, 0.0, 0.0])
    
    def intersection_volume(self) -> float:
        """计算当前相对位置下的交集体积（使用半空间交集）"""
        from scipy.spatial import HalfspaceIntersection
        verts1 = self.s1.get_transformed_vertices()
        verts2 = self.s2.get_transformed_vertices()
        try:
            hull1 = ConvexHull(verts1)
            hull2 = ConvexHull(verts2)
            A1, b1 = hull1.equations[:, :3], hull1.equations[:, 3]
            A2, b2 = hull2.equations[:, :3], hull2.equations[:, 3]
            A = np.vstack([A1, A2])
            b = np.hstack([b1, b2])
            interior = (np.mean(verts1, axis=0) + np.mean(verts2, axis=0)) / 2
            hs = HalfspaceIntersection(A, b, interior)
            return ConvexHull(hs.intersections).volume
        except:
            return 0.0
    
    def combat_intensity(self) -> float:
        vol1 = self.s1.geometry.get_statistics()['Volume']
        vol2 = self.s2.geometry.get_statistics()['Volume']
        if vol1 == 0 or vol2 == 0:
            return 0.0
        inter = self.intersection_volume()
        union = vol1 + vol2 - inter
        return inter / union if union > 0 else 0.0
    
    def visualize_dynamic(self):
        """打开交互窗口，可通过滑块控制相对距离和旋转角度"""
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.25)
        
        # 滑块：距离
        ax_dist = plt.axes([0.2, 0.1, 0.6, 0.03])
        self.slider_dist = Slider(ax_dist, 'Distance', 0.5, 8.0, valinit=3.0)
        self.slider_dist.on_changed(self._update)
        
        # 滑块：角度（绕Y轴旋转第二个替身）
        ax_angle = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider_angle = Slider(ax_angle, 'Angle (deg)', 0, 360, valinit=0)
        self.slider_angle.on_changed(self._update)
        
        self._update(None)
        plt.show()
    
    def _update(self, val):
        self.ax.clear()
        dist = self.slider_dist.val
        angle = np.deg2rad(self.slider_angle.val)
        self.update_positions_by_range(dist)
        # 旋转第二个替身
        rot = R.from_euler('y', angle).as_matrix()
        self.s2.position = np.array([dist/2, 0, 0])  # 重置位置再旋转？
        # 更自然：保持距离但允许旋转
        self.s2.position = np.array([dist/2, 0, 0])
        # 实际上替身本身不旋转，只是位置偏移，这里我们只旋转位置向量
        self.s2.position = rot @ self.s2.position
        
        # 绘制两个替身
        for stand, alpha in [(self.s1, 0.6), (self.s2, 0.6)]:
            verts = stand.get_transformed_vertices()
            poly = Poly3DCollection([verts[f] for f in stand.geometry.faces], alpha=alpha,
                                    facecolor=stand.color, edgecolor='k', linewidth=0.5)
            self.ax.add_collection3d(poly)
            # 边
            for e in stand.geometry.edges:
                self.ax.plot3D(*zip(verts[e[0]], verts[e[1]]), 'k-', lw=0.8)
        
        # 计算并显示交集体积/强度
        inter_vol = self.intersection_volume()
        intensity = self.combat_intensity()
        self.ax.set_title(f"Dynamic Combat: Distance={dist:.2f}  Angle={self.slider_angle.val:.0f}°\n"
                          f"Intersection Volume: {inter_vol:.3f}  Intensity: {intensity:.3f}")
        
        # 坐标轴范围
        all_verts = np.vstack([self.s1.get_transformed_vertices(), self.s2.get_transformed_vertices()])
        max_range = np.max(np.abs(all_verts)) + 1.0
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([-max_range, max_range])
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        self.fig.canvas.draw_idle()

# -------------------- 2. 高级进化动画（形状匹配）--------------------
class AdvancedEvolutionAnimator:
    """
    使用迭代最近点(ICP)建立顶点对应，并通过球面插值实现平滑拓扑过渡
    """
    @staticmethod
    def establish_correspondence(verts_low: np.ndarray, verts_high: np.ndarray) -> np.ndarray:
        """使用KDTree为低模每个顶点找高模最近点，返回对应索引"""
        tree = KDTree(verts_high)
        _, idx = tree.query(verts_low)
        return idx
    
    @staticmethod
    def spherical_interpolation(v1: np.ndarray, v2: np.ndarray, t: float, radius: float) -> np.ndarray:
        """在球面上进行插值（保持半径）"""
        # 归一化到球面
        if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
            return (1-t)*v1 + t*v2
        v1_norm = v1 / np.linalg.norm(v1) * radius
        v2_norm = v2 / np.linalg.norm(v2) * radius
        # 球面线性插值 (slerp)
        dot = np.dot(v1_norm, v2_norm) / (radius*radius)
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot) * t
        v3 = v2_norm - v1_norm * dot
        v3 = v3 / np.linalg.norm(v3)
        return v1_norm * np.cos(theta) + v3 * np.sin(theta) * radius
    
    @staticmethod
    def interpolate_geometries(geo_low: PolyhedronGeometry, geo_high: PolyhedronGeometry, t: float):
        """
        在两个几何体之间插值。低模顶点通过对应关系映射到高模，插值后重建凸包。
        返回 vertices, faces
        """
        verts_low = np.array(geo_low.vertices)
        verts_high = np.array(geo_high.vertices)
        radius = geo_low.radius * (1-t) + geo_high.radius * t
        
        # 建立对应
        corr_idx = AdvancedEvolutionAnimator.establish_correspondence(verts_low, verts_high)
        target_verts = verts_high[corr_idx]
        
        # 插值每个顶点
        interp_verts = []
        for v_low, v_target in zip(verts_low, target_verts):
            interp_v = AdvancedEvolutionAnimator.spherical_interpolation(v_low, v_target, t, radius)
            interp_verts.append(interp_v)
        interp_verts = np.array(interp_verts)
        
        # 添加高模独有的顶点（当t较大时逐渐出现）
        if t > 0.5:
            # 简单处理：将高模独有顶点按t混合加入
            extra_verts = []
            for i, v in enumerate(verts_high):
                if i not in corr_idx:
                    alpha = (t - 0.5) * 2  # 0.5->0, 1.0->1
                    extra_verts.append(v * alpha)
            if extra_verts:
                interp_verts = np.vstack([interp_verts, extra_verts])
        
        # 重建凸包
        try:
            hull = ConvexHull(interp_verts)
            final_verts = interp_verts[hull.vertices]
            faces = hull.simplices
        except:
            # 退化为原始几何
            final_verts = interp_verts
            faces = geo_low.faces if t < 0.5 else geo_high.faces
        
        return final_verts, faces
    
    @staticmethod
    def animate_evolution(stand: Stand, target_potential: StatRank, frames=40):
        """展示平滑进化动画"""
        stats_low = copy.deepcopy(stand.stats)
        stats_high = copy.deepcopy(stand.stats)
        stats_high.potential = target_potential
        geo_low = PolyhedronGeometry(stats_low)
        geo_high = PolyhedronGeometry(stats_high)
        
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            t = frame / (frames-1)
            verts, faces = AdvancedEvolutionAnimator.interpolate_geometries(geo_low, geo_high, t)
            poly = Poly3DCollection([verts[f] for f in faces], alpha=0.6, color=stand.color)
            ax.add_collection3d(poly)
            # 边
            for f in faces:
                for i in range(len(f)):
                    v1, v2 = verts[f[i]], verts[f[(i+1)%len(f)]]
                    ax.plot3D(*zip(v1, v2), 'k-', lw=0.5)
            ax.scatter(verts[:,0], verts[:,1], verts[:,2], c='r', s=10)
            lim = max(geo_low.radius, geo_high.radius) * 1.8
            ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([-lim, lim])
            ax.set_title(f"Evolution: Potential {stand.stats.potential.name} → {target_potential.name}\nFrame {frame+1}/{frames}")
        
        ani = FuncAnimation(fig, update, frames=frames, interval=80, repeat=False)
        plt.show()
        return ani

# -------------------- 3. 导出增强（材质+纹理坐标）--------------------
class EnhancedStandExporter:
    @staticmethod
    def compute_sphere_uv(vertices: np.ndarray) -> List[Tuple[float, float]]:
        """球面映射UV坐标"""
        uvs = []
        for v in vertices:
            norm = np.linalg.norm(v)
            if norm < 1e-6:
                uvs.append((0.5, 0.5))
                continue
            x, y, z = v / norm
            u = 0.5 + np.arctan2(x, z) / (2 * np.pi)
            v_ = 0.5 + np.arcsin(y) / np.pi
            uvs.append((u, v_))
        return uvs
    
    @staticmethod
    def to_obj_with_mtl(stand: Stand, filename: str, mtl_name: str = None):
        """导出OBJ+MTL材质文件"""
        base = os.path.splitext(filename)[0]
        if mtl_name is None:
            mtl_name = base + ".mtl"
        else:
            mtl_name = os.path.basename(mtl_name)
        
        verts = np.array(stand.geometry.vertices)
        faces = stand.geometry.faces
        uvs = EnhancedStandExporter.compute_sphere_uv(verts)
        
        # 写OBJ
        with open(filename, 'w') as f:
            f.write(f"# OBJ exported from JOJO Stand Engine\n")
            f.write(f"# Stand: {stand.name} | Master: {stand.master}\n")
            f.write(f"mtllib {mtl_name}\n")
            f.write(f"o {stand.name}\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for uv in uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
            f.write(f"usemtl {stand.name}_material\n")
            for face in faces:
                f.write("f " + " ".join(f"{idx+1}/{idx+1}" for idx in face) + "\n")
        
        # 写MTL
        with open(base + ".mtl", 'w') as f:
            f.write(f"newmtl {stand.name}_material\n")
            f.write(f"Kd {stand.color[0]:.3f} {stand.color[1]:.3f} {stand.color[2]:.3f}\n")
            f.write("Ks 0.5 0.5 0.5\n")
            f.write("Ns 96.0\n")
            f.write("illum 2\n")
        print(f"Exported OBJ+MTL to {filename} and {base}.mtl")
    
    @staticmethod
    def to_stl_color(stand: Stand, filename: str):
        """导出带颜色的STL（ASCII格式，颜色通过扩展名支持有限，此处仅标准STL）"""
        # STL不支持材质，但我们可以输出带颜色的ASCII STL（非标准但某些软件支持）
        verts = np.array(stand.geometry.vertices)
        faces = stand.geometry.faces
        with open(filename, 'w') as f:
            f.write(f"solid {stand.name}\n")
            for face in faces:
                if len(face) >= 3:
                    v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
                    normal = np.cross(v1-v0, v2-v0)
                    norm = np.linalg.norm(normal)
                    if norm > 0:
                        normal = normal / norm
                    f.write(f"facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                    f.write("  outer loop\n")
                    f.write(f"    vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
                    f.write(f"    vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
                    f.write(f"    vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
                    f.write("  endloop\nendfacet\n")
            f.write(f"endsolid {stand.name}\n")
        print(f"Exported STL to {filename}")

# -------------------- 4. 创建器升级（保存/加载配置）--------------------
class AdvancedStandCreatorApp:
    """交互式创建器，支持保存/加载JSON配置"""
    def __init__(self):
        self.fig = plt.figure(figsize=(14, 9))
        self.ax3d = self.fig.add_axes([0.05, 0.25, 0.55, 0.7], projection='3d')
        self.sliders = {}
        stat_names = ['Power', 'Speed', 'Range', 'Durability', 'Precision', 'Potential']
        y_start = 0.85
        for i, name in enumerate(stat_names):
            ax = self.fig.add_axes([0.65, y_start - i*0.08, 0.25, 0.03])
            slider = Slider(ax, name, 1, 5, valinit=3, valstep=1)
            slider.on_changed(self.update)
            self.sliders[name] = slider
        
        # 按钮
        btn_save_ax = self.fig.add_axes([0.65, 0.15, 0.1, 0.04])
        self.btn_save = Button(btn_save_ax, 'Save')
        self.btn_save.on_clicked(self.save_stand)
        
        btn_load_ax = self.fig.add_axes([0.77, 0.15, 0.1, 0.04])
        self.btn_load = Button(btn_load_ax, 'Load')
        self.btn_load.on_clicked(self.load_stand)
        
        btn_rand_ax = self.fig.add_axes([0.65, 0.08, 0.1, 0.04])
        self.btn_rand = Button(btn_rand_ax, 'Random')
        self.btn_rand.on_clicked(self.randomize)
        
        # 信息文本
        self.info_ax = self.fig.add_axes([0.65, 0.22, 0.25, 0.1])
        self.info_ax.axis('off')
        self.info_text = self.info_ax.text(0.5, 0.5, '', ha='center', va='center', fontsize=9,
                                           transform=self.info_ax.transAxes)
        
        self.current_stand = None
        self.update(None)
        plt.show()
    
    def get_stats_from_sliders(self):
        vals = {name: StatRank(int(slider.val)) for name, slider in self.sliders.items()}
        return StandStats(**vals)
    
    def update(self, val):
        stats = self.get_stats_from_sliders()
        self.current_stand = Stand("Custom Stand", "User", stats)
        self.draw_stand()
        self.update_info()
    
    def draw_stand(self):
        self.ax3d.clear()
        verts = np.array(self.current_stand.geometry.vertices)
        poly = Poly3DCollection([verts[f] for f in self.current_stand.geometry.faces],
                                alpha=0.5, facecolor=self.current_stand.color, edgecolor='k', linewidth=0.5)
        self.ax3d.add_collection3d(poly)
        for e in self.current_stand.geometry.edges:
            self.ax3d.plot3D(*zip(verts[e[0]], verts[e[1]]), 'k-', lw=0.8)
        self.ax3d.scatter(verts[:,0], verts[:,1], verts[:,2], c='r', s=20)
        lim = self.current_stand.geometry.radius * 1.8
        self.ax3d.set_xlim([-lim, lim]); self.ax3d.set_ylim([-lim, lim]); self.ax3d.set_zlim([-lim, lim])
        self.ax3d.set_title("Custom Stand Creator")
        self.fig.canvas.draw_idle()
    
    def update_info(self):
        s = self.current_stand.geometry.get_statistics()
        text = f"Volume: {s['Volume']:.3f}\nEuler: {s['Euler_Characteristic']}\nRadius: {s['Radius']:.2f}"
        self.info_text.set_text(text)
    
    def randomize(self, event):
        for slider in self.sliders.values():
            slider.set_val(np.random.randint(1,6))
        self.update(None)
    
    def save_stand(self, event):
        data = self.current_stand.to_dict()
        filename = f"{self.current_stand.name.replace(' ', '_')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Stand saved to {filename}")
    
    def load_stand(self, event):
        # 简单演示：列出当前目录JSON文件供选择（此处简化为固定文件名示例）
        # 实际应用可添加文件对话框，这里为演示使用 tkinter 简单对话框
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'r') as f:
                data = json.load(f)
            loaded_stand = Stand.from_dict(data)
            # 更新滑块
            for name, rank in loaded_stand.stats.__dict__.items():
                self.sliders[name.capitalize()].set_val(rank.value)
            self.current_stand = loaded_stand
            self.draw_stand()
            self.update_info()
            print(f"Loaded stand from {file_path}")

# -------------------- 演示所有新功能 --------------------
if __name__ == "__main__":
    # 创建示例替身
    sp_stats = StandStats(StatRank.A, StatRank.A, StatRank.C, StatRank.A, StatRank.A, StatRank.B)
    tw_stats = StandStats(StatRank.A, StatRank.A, StatRank.C, StatRank.A, StatRank.B, StatRank.B)
    ge_stats = StandStats(StatRank.C, StatRank.A, StatRank.E, StatRank.D, StatRank.C, StatRank.A)
    
    star_platinum = Stand("Star Platinum", "Jotaro", sp_stats, "Time stop")
    the_world = Stand("The World", "DIO", tw_stats, "Time stop")
    gold_exp = Stand("Gold Experience", "Giorno", ge_stats, "Life giver")
    
    print("=== 1. 动态战斗模拟（带位置偏移滑块）===")
    combat = DynamicCombatSimulator(star_platinum, the_world)
    combat.visualize_dynamic()  # 打开交互窗口，可调节距离和角度
    
    print("\n=== 2. 高级进化动画（形状匹配）===")
    AdvancedEvolutionAnimator.animate_evolution(gold_exp, StatRank.INFINITE, frames==60)
    
    print("\n=== 3. 导出增强（材质+纹理）===")
    EnhancedStandExporter.to_obj_with_mtl(star_platinum, "star_platinum.obj")
    EnhancedStandExporter.to_stl_color(the_world, "the_world.stl")
    
    print("\n=== 4. 交互式创建器（支持保存/加载）===")
    app = AdvancedStandCreatorApp()
