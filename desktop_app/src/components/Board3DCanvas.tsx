import { useEffect, useMemo, useRef, useState, type ComponentRef } from "react";
import { Canvas, useThree, type ThreeEvent } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import { CanvasTexture, DoubleSide, SRGBColorSpace } from "three";

import {
  BOARD_SIZE,
  CELL_GAP,
  MAX_LAYERS,
  boardToWorld,
  columnKey,
  coordinateKey,
  createColumnGuides,
  isIntentionalBoardClick,
  layerGap,
  legalMovesByColumn,
  moveMatchesSlice,
  pieceRenderMode,
} from "../board3dModel";
import type { CameraCommand, GameState, HintResult, LayerSpacing, Move, PieceFocus, Player, SliceSelection } from "../types";

const RED = "#ff5263";
const BLUE = "#478fff";
const CYAN = "#50d7dc";
const GOLD = "#ffd65a";
const MIN_CAMERA_ZOOM = 28;
const MAX_CAMERA_ZOOM = 105;

interface Props {
  state: GameState;
  hint: HintResult | null;
  moveLocked: boolean;
  compactLayout: boolean;
  showCoordinateLabels: boolean;
  pieceFocus: PieceFocus;
  showColumnGuides: boolean;
  slicePickerEnabled: boolean;
  sliceSelection: SliceSelection | null;
  layerSpacing: LayerSpacing;
  cameraCommand: CameraCommand;
  onMove: (move: Move) => void;
  onHoverMove: (move: Move | null) => void;
  onSliceSelection: (selection: SliceSelection) => void;
}

function cameraZoom(command: CameraCommand, spacing: LayerSpacing, compactLayout: boolean): number {
  if (compactLayout) {
    if (command.preset === "top") return 38;
    if (spacing === "expanded") return command.preset === "front" ? 30 : 22;
    return command.preset === "front" ? 36 : 26;
  }
  if (command.preset === "top") return 64;
  if (spacing === "expanded") return command.preset === "front" ? 40 : 44;
  return 52;
}

function CameraRig({
  command,
  spacing,
  sliceActive,
  compactLayout,
}: {
  command: CameraCommand;
  spacing: LayerSpacing;
  sliceActive: boolean;
  compactLayout: boolean;
}) {
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  const sliceActiveRef = useRef(sliceActive);
  const zoomBeforeSlice = useRef<number | null>(null);
  const { camera, invalidate } = useThree();

  useEffect(() => {
    const positions = {
      isometric: [8.6, 7.6, 8.6],
      front: [0, 1.2, 12],
      top: [0, 12, 0.001],
    } as const;
    const [x, y, z] = positions[command.preset];
    camera.position.set(x, y, z);
    camera.up.set(0, command.preset === "top" ? 0 : 1, command.preset === "top" ? -1 : 0);
    if ("zoom" in camera) {
      const baseZoom = cameraZoom(command, spacing, compactLayout);
      zoomBeforeSlice.current = sliceActive ? baseZoom : null;
      camera.zoom = baseZoom * (sliceActive ? 0.9 : 1);
    }
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
    controls.current?.target.set(0, 0, 0);
    controls.current?.update();
    sliceActiveRef.current = sliceActive;
    invalidate();
    const frame = requestAnimationFrame(() => invalidate());
    return () => cancelAnimationFrame(frame);
  }, [camera, command, compactLayout, invalidate, spacing]); // slice state is fitted separately without resetting the orbit

  useEffect(() => {
    if (sliceActiveRef.current === sliceActive || !("zoom" in camera)) return;
    if (sliceActive) {
      zoomBeforeSlice.current = camera.zoom;
      camera.zoom = Math.max(MIN_CAMERA_ZOOM, Math.min(MAX_CAMERA_ZOOM, camera.zoom * 0.9));
    } else {
      camera.zoom = Math.max(MIN_CAMERA_ZOOM, Math.min(MAX_CAMERA_ZOOM, zoomBeforeSlice.current ?? camera.zoom / 0.9));
      zoomBeforeSlice.current = null;
    }
    camera.updateProjectionMatrix();
    controls.current?.update();
    sliceActiveRef.current = sliceActive;
    invalidate();
    const frame = requestAnimationFrame(() => invalidate());
    return () => cancelAnimationFrame(frame);
  }, [camera, invalidate, sliceActive]);

  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enablePan={false}
      enableDamping
      dampingFactor={0.08}
      minPolarAngle={0.12}
      maxPolarAngle={Math.PI / 2 - 0.04}
      minZoom={compactLayout ? 16 : MIN_CAMERA_ZOOM}
      maxZoom={MAX_CAMERA_ZOOM}
    />
  );
}

function BoardLattice({ spacing, guidesVisible }: { spacing: LayerSpacing; guidesVisible: boolean }) {
  const half = BOARD_SIZE * CELL_GAP / 2;
  const gridOpacity = guidesVisible ? 0.075 : 0.13;
  const lines = useMemo(() => {
    const result: Array<{ key: string; points: Array<[number, number, number]> }> = [];
    for (let layer = 0; layer < MAX_LAYERS; layer += 1) {
      const y = boardToWorld(layer, 0, 0, spacing)[1];
      for (let index = 0; index <= BOARD_SIZE; index += 1) {
        const offset = -half + index * CELL_GAP;
        result.push({ key: `x:${layer}:${index}`, points: [[-half, y, offset], [half, y, offset]] });
        result.push({ key: `z:${layer}:${index}`, points: [[offset, y, -half], [offset, y, half]] });
      }
    }
    return result;
  }, [half, spacing]);

  return (
    <group>
      {lines.map((line) => (
        <Line key={line.key} points={line.points} color="#6b9ab7" lineWidth={1} transparent opacity={gridOpacity} />
      ))}
    </group>
  );
}

function OutlinePiece({ color }: { color: string }) {
  const rotations: Array<[number, number, number]> = [
    [0, 0, 0],
    [Math.PI / 2, 0, 0],
    [0, Math.PI / 2, 0],
  ];
  return (
    <group>
      {rotations.map((rotation, index) => (
        <mesh rotation={rotation} key={index}>
          <torusGeometry args={[0.335, 0.014, 8, 44]} />
          <meshBasicMaterial color={color} transparent opacity={0.68} depthWrite={false} />
        </mesh>
      ))}
    </group>
  );
}

function BoardPieces({ state, focus, spacing, selection }: { state: GameState; focus: PieceFocus; spacing: LayerSpacing; selection: SliceSelection | null }) {
  const winning = useMemo(() => new Set(state.winning_line.map(coordinateKey)), [state.winning_line]);
  const last = state.last_move ? coordinateKey(state.last_move) : "";

  return (
    <group>
      {state.board.flatMap((layer, layerIndex) => layer.flatMap((row, rowIndex) => row.map((value, colIndex) => {
        if (value !== 1 && value !== -1) return null;
        if (selection && !moveMatchesSlice({ layer: layerIndex, row: rowIndex, col: colIndex }, selection)) return null;
        const key = `${layerIndex}:${rowIndex}:${colIndex}`;
        const player = value as Player;
        const mode = pieceRenderMode(player, focus, winning.has(key));
        const color = mode === "winning" ? GOLD : player === 1 ? RED : BLUE;
        const position = boardToWorld(layerIndex, rowIndex, colIndex, spacing);
        return (
          <group position={position} key={key}>
            {mode === "outline" ? (
              <OutlinePiece color={color} />
            ) : (
              <mesh>
                <sphereGeometry args={[0.36, 28, 20]} />
                <meshStandardMaterial
                  color={color}
                  roughness={0.34}
                  metalness={0.12}
                  emissive={color}
                  emissiveIntensity={mode === "winning" ? 0.48 : 0.14}
                />
              </mesh>
            )}
            {key === last && (
              <mesh rotation={[Math.PI / 2, 0, 0]}>
                <torusGeometry args={[0.45, 0.025, 10, 48]} />
                <meshBasicMaterial color="#f5fbff" transparent opacity={0.95} />
              </mesh>
            )}
          </group>
        );
      })))}
    </group>
  );
}

function WinningLine({ state, spacing, selection }: { state: GameState; spacing: LayerSpacing; selection: SliceSelection | null }) {
  if (state.winning_line.length < 2) return null;
  if (selection && !state.winning_line.every((move) => moveMatchesSlice(move, selection))) return null;
  const points = state.winning_line.map((move) => boardToWorld(move.layer, move.row, move.col, spacing));
  return <Line points={points} color={GOLD} lineWidth={5} transparent opacity={0.96} />;
}

interface ColumnsProps {
  moves: Move[];
  moveLocked: boolean;
  showGuides: boolean;
  spacing: LayerSpacing;
  selection: SliceSelection | null;
  hoveredKey: string;
  onHover: (move: Move | null) => void;
  onMove: (move: Move) => void;
}

function InteractiveColumns(props: ColumnsProps) {
  const guides = useMemo(createColumnGuides, []);
  const visibleGuides = useMemo(() => guides.filter((guide) => (
    !props.selection
    || props.selection.axis === "layer"
    || (props.selection.axis === "col" ? guide.col : guide.row) === props.selection.index
  )), [guides, props.selection]);
  const legal = useMemo(() => legalMovesByColumn(props.selection
    ? props.moves.filter((move) => moveMatchesSlice(move, props.selection!))
    : props.moves), [props.moves, props.selection]);
  const bottomY = boardToWorld(0, 0, 0, props.spacing)[1] - 0.46;
  const topY = boardToWorld(MAX_LAYERS - 1, 0, 0, props.spacing)[1] + 0.46;
  const guideHeight = topY - bottomY;

  return (
    <group>
      {visibleGuides.map((guide) => {
        const move = legal.get(guide.key) ?? null;
        const [x, , z] = boardToWorld(0, guide.row, guide.col, props.spacing);
        const hovered = props.hoveredKey === guide.key;
        const activate = () => {
          if (move && !props.moveLocked) props.onMove(move);
        };
        return (
          <group key={guide.key}>
            {props.showGuides && (
              <mesh
                position={[x, (topY + bottomY) / 2, z]}
                onPointerOver={(event) => { event.stopPropagation(); props.onHover(move); }}
                onPointerOut={(event) => { event.stopPropagation(); props.onHover(null); }}
                onClick={(event) => { event.stopPropagation(); if (isIntentionalBoardClick(event.delta)) activate(); }}
              >
                <cylinderGeometry args={[hovered ? 0.04 : 0.024, hovered ? 0.04 : 0.024, guideHeight, 10]} />
                <meshBasicMaterial color={CYAN} transparent opacity={hovered ? 0.7 : 0.2} depthWrite={false} />
              </mesh>
            )}
            <mesh
              position={[x, bottomY + 0.02, z]}
              rotation={[-Math.PI / 2, 0, 0]}
              onPointerOver={(event) => { event.stopPropagation(); props.onHover(move); }}
              onPointerOut={(event) => { event.stopPropagation(); props.onHover(null); }}
              onClick={(event) => { event.stopPropagation(); if (isIntentionalBoardClick(event.delta)) activate(); }}
            >
              <planeGeometry args={[CELL_GAP * 0.82, CELL_GAP * 0.82]} />
              <meshBasicMaterial
                color={move ? CYAN : "#31485d"}
                transparent
                opacity={hovered && move ? 0.3 : move ? 0.075 : 0.025}
                depthWrite={false}
              />
            </mesh>
            {hovered && move && (
              <mesh position={boardToWorld(move.layer, move.row, move.col, props.spacing)}>
                <sphereGeometry args={[0.39, 24, 16]} />
                <meshStandardMaterial
                  color={CYAN}
                  emissive={CYAN}
                  emissiveIntensity={0.4}
                  transparent
                  opacity={props.moveLocked ? 0.22 : 0.45}
                  depthWrite={false}
                />
              </mesh>
            )}
          </group>
        );
      })}
    </group>
  );
}

interface AxisLabelProps {
  text: string;
  position: [number, number, number];
  selectable: boolean;
  selected: boolean;
  onSelect: () => void;
}

function AxisLabel({ text, position, selectable, selected, onSelect }: AxisLabelProps) {
  const cursorTarget = useRef<HTMLElement | null>(null);
  const texture = useMemo(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 128;
    canvas.height = 64;
    const context = canvas.getContext("2d");
    if (context) {
      context.fillStyle = selected ? "rgba(49, 31, 72, 0.96)" : "rgba(5, 19, 32, 0.9)";
      context.fillRect(4, 4, 120, 56);
      context.strokeStyle = selected
        ? "rgba(202, 160, 255, 0.98)"
        : selectable
          ? "rgba(181, 133, 255, 0.9)"
          : "rgba(80, 215, 220, 0.5)";
      context.lineWidth = selected ? 5 : selectable ? 4 : 3;
      context.strokeRect(5.5, 5.5, 117, 53);
      context.fillStyle = selected ? "#f3e8ff" : selectable ? "#eadbff" : "#9af1f3";
      context.font = "700 30px Segoe UI, sans-serif";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(text, 64, 33);
    }
    const result = new CanvasTexture(canvas);
    result.colorSpace = SRGBColorSpace;
    return result;
  }, [selectable, selected, text]);

  useEffect(() => () => {
    texture.dispose();
    if (cursorTarget.current) cursorTarget.current.style.cursor = "";
  }, [texture]);

  const pointerProps = selectable ? {
    onPointerOver: (event: ThreeEvent<PointerEvent>) => {
      event.stopPropagation();
      cursorTarget.current = event.nativeEvent.target as HTMLElement;
      cursorTarget.current.style.cursor = "pointer";
    },
    onPointerOut: (event: ThreeEvent<PointerEvent>) => {
      event.stopPropagation();
      (event.nativeEvent.target as HTMLElement).style.cursor = "";
      cursorTarget.current = null;
    },
    onClick: (event: ThreeEvent<MouseEvent>) => {
      event.stopPropagation();
      if (isIntentionalBoardClick(event.delta)) onSelect();
    },
  } : {};

  return (
    <sprite position={position} scale={selected ? [0.7, 0.35, 1] : [0.62, 0.31, 1]} {...pointerProps}>
      <spriteMaterial map={texture} transparent depthTest={false} depthWrite={false} />
    </sprite>
  );
}

interface CoordinateLabelsProps {
  spacing: LayerSpacing;
  visible: boolean;
  selectable: boolean;
  selection: SliceSelection | null;
  onSelect: (selection: SliceSelection) => void;
}

function CoordinateLabels({ spacing, visible, selectable, selection, onSelect }: CoordinateLabelsProps) {
  if (!visible) return null;
  const half = BOARD_SIZE * CELL_GAP / 2;
  const y = boardToWorld(0, 0, 0, spacing)[1] - 0.53;
  return (
    <group>
      {Array.from({ length: BOARD_SIZE }, (_, col) => {
        const [x] = boardToWorld(0, 0, col, spacing);
        return (
          <AxisLabel
            key={`c${col}`}
            text={`C${col + 1}`}
            position={[x, y, half + 0.28]}
            selectable={selectable}
            selected={selection?.axis === "col" && selection.index === col}
            onSelect={() => onSelect({ axis: "col", index: col })}
          />
        );
      })}
      {Array.from({ length: BOARD_SIZE }, (_, row) => {
        const [, , z] = boardToWorld(0, row, 0, spacing);
        return (
          <AxisLabel
            key={`r${row}`}
            text={`R${row + 1}`}
            position={[-half - 0.28, y, z]}
            selectable={selectable}
            selected={selection?.axis === "row" && selection.index === row}
            onSelect={() => onSelect({ axis: "row", index: row })}
          />
        );
      })}
      {Array.from({ length: MAX_LAYERS }, (_, layer) => {
        const layerY = boardToWorld(layer, 0, 0, spacing)[1];
        return (
          <AxisLabel
            key={`f${layer}`}
            text={`F${layer + 1}`}
            position={[half + 0.3, layerY, -half - 0.18]}
            selectable={selectable}
            selected={selection?.axis === "layer" && selection.index === layer}
            onSelect={() => onSelect({ axis: "layer", index: layer })}
          />
        );
      })}
    </group>
  );
}

function SlicePlane({ selection, spacing }: { selection: SliceSelection; spacing: LayerSpacing }) {
  const half = BOARD_SIZE * CELL_GAP / 2;
  const bottom = boardToWorld(0, 0, 0, spacing)[1] - 0.5;
  const top = boardToWorld(MAX_LAYERS - 1, 0, 0, spacing)[1] + 0.5;
  const height = top - bottom;
  const centerY = (top + bottom) / 2;
  const planeColor = "#b985ff";

  if (selection.axis === "col") {
    const x = boardToWorld(0, 0, selection.index, spacing)[0];
    const outline: Array<[number, number, number]> = [[x, bottom, -half], [x, bottom, half], [x, top, half], [x, top, -half], [x, bottom, -half]];
    return (
      <group>
        <mesh position={[x, centerY, 0]} rotation={[0, Math.PI / 2, 0]}>
          <planeGeometry args={[BOARD_SIZE * CELL_GAP, height]} />
          <meshBasicMaterial color={planeColor} transparent opacity={0.055} depthWrite={false} side={DoubleSide} />
        </mesh>
        <Line points={outline} color={planeColor} lineWidth={2} transparent opacity={0.62} />
      </group>
    );
  }

  if (selection.axis === "row") {
    const z = boardToWorld(0, selection.index, 0, spacing)[2];
    const outline: Array<[number, number, number]> = [[-half, bottom, z], [half, bottom, z], [half, top, z], [-half, top, z], [-half, bottom, z]];
    return (
      <group>
        <mesh position={[0, centerY, z]}>
          <planeGeometry args={[BOARD_SIZE * CELL_GAP, height]} />
          <meshBasicMaterial color={planeColor} transparent opacity={0.055} depthWrite={false} side={DoubleSide} />
        </mesh>
        <Line points={outline} color={planeColor} lineWidth={2} transparent opacity={0.62} />
      </group>
    );
  }

  const layerY = boardToWorld(selection.index, 0, 0, spacing)[1];
  const outline: Array<[number, number, number]> = [[-half, layerY, -half], [half, layerY, -half], [half, layerY, half], [-half, layerY, half], [-half, layerY, -half]];
  return (
    <group>
      <mesh position={[0, layerY, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[BOARD_SIZE * CELL_GAP, BOARD_SIZE * CELL_GAP]} />
        <meshBasicMaterial color={planeColor} transparent opacity={0.055} depthWrite={false} side={DoubleSide} />
      </mesh>
      <Line points={outline} color={planeColor} lineWidth={2} transparent opacity={0.62} />
    </group>
  );
}

function Scene(props: Props & { hoveredMove: Move | null; setHoveredMove: (move: Move | null) => void }) {
  const hoveredKey = props.hoveredMove ? columnKey(props.hoveredMove.row, props.hoveredMove.col) : "";
  const hintVisible = props.hint
    && props.state.board[props.hint.move.layer]?.[props.hint.move.row]?.[props.hint.move.col] === 0
    && (!props.sliceSelection || moveMatchesSlice(props.hint.move, props.sliceSelection));

  return (
    <>
      <ambientLight intensity={1.3} />
      <directionalLight position={[7, 11, 8]} intensity={2.1} color="#dff7ff" />
      <directionalLight position={[-7, 2, -5]} intensity={0.7} color="#5478b8" />
      <CameraRig
        command={props.cameraCommand}
        spacing={props.layerSpacing}
        sliceActive={Boolean(props.sliceSelection)}
        compactLayout={props.compactLayout}
      />
      <BoardLattice spacing={props.layerSpacing} guidesVisible={props.showColumnGuides} />
      <CoordinateLabels
        spacing={props.layerSpacing}
        visible={props.showCoordinateLabels && (props.showColumnGuides || props.slicePickerEnabled)}
        selectable={props.slicePickerEnabled}
        selection={props.sliceSelection}
        onSelect={props.onSliceSelection}
      />
      {props.sliceSelection && <SlicePlane selection={props.sliceSelection} spacing={props.layerSpacing} />}
      <InteractiveColumns
        moves={props.state.legal_moves}
        moveLocked={props.moveLocked}
        showGuides={props.showColumnGuides}
        spacing={props.layerSpacing}
        selection={props.sliceSelection}
        hoveredKey={hoveredKey}
        onHover={(move) => { props.setHoveredMove(move); props.onHoverMove(move); }}
        onMove={props.onMove}
      />
      <BoardPieces state={props.state} focus={props.pieceFocus} spacing={props.layerSpacing} selection={props.sliceSelection} />
      <WinningLine state={props.state} spacing={props.layerSpacing} selection={props.sliceSelection} />
      {hintVisible && props.hint && (
        <mesh position={boardToWorld(props.hint.move.layer, props.hint.move.row, props.hint.move.col, props.layerSpacing)}>
          <sphereGeometry args={[0.405, 24, 16]} />
          <meshStandardMaterial color={GOLD} emissive={GOLD} emissiveIntensity={0.52} transparent opacity={0.4} depthWrite={false} />
        </mesh>
      )}
    </>
  );
}

export function Board3DCanvas(props: Props) {
  const [hoveredMove, setHoveredMove] = useState<Move | null>(null);

  useEffect(() => {
    setHoveredMove(null);
    props.onHoverMove(null);
  }, [
    props.showColumnGuides,
    props.slicePickerEnabled,
    props.sliceSelection?.axis,
    props.sliceSelection?.index,
    props.state.revision,
  ]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <Canvas
      orthographic
      frameloop="demand"
      dpr={[1, 1.5]}
      camera={{ position: [8.6, 7.6, 8.6], zoom: props.compactLayout ? 26 : 52, near: 0.1, far: 100 }}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      onPointerMissed={() => { setHoveredMove(null); props.onHoverMove(null); }}
      onCreated={({ gl }) => gl.setClearColor(0x000000, 0)}
    >
      <Scene {...props} hoveredMove={hoveredMove} setHoveredMove={setHoveredMove} />
    </Canvas>
  );
}
