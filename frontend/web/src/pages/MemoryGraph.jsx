import React, { useState, useEffect, useCallback } from 'react';
import ReactFlow, {
  Background, Controls, MiniMap,
  useNodesState, useEdgesState,
  Handle, Position,
} from 'reactflow';
import dagre from 'dagre';
import 'reactflow/dist/style.css';
import { useWorkspace } from '../context/WorkspaceContext';
import { listMemoryNodes, deleteMemoryNode, listRuns } from '../lib/api';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { ScrollArea } from '../components/ui/scroll-area';
import { Separator } from '../components/ui/separator';
import {
  RefreshCw,
  Trash2,
  GitBranch,
  Loader2,
  ArrowRight,
  FileText,
  Wrench,
  CheckSquare,
  Cpu,
  Network,
  Brain,
  Terminal,
  BookOpen,
  Layers,
  FolderTree,
  ScrollText,
  FileCode2,
  Archive,
  Trophy,
} from 'lucide-react';
import { toast } from 'sonner';

// Node icon mapping (includes MemPalace store-derived types)
const nodeIcons = {
  concept: FileText,
  entity: Wrench,
  task: CheckSquare,
  decision: Cpu,
  reference: FileText,
  subtask_result: CheckSquare,
  lesson: Brain,
  knowledge: FileText,
  scratchpad: FileText,
  signal: Cpu,
  gap: Network,
  verification: CheckSquare,
  run_result: Terminal,
  source: FolderTree,
  empty_source: Archive,
  prompt: BookOpen,
  workspace_prompt: BookOpen,
  prompt_snapshot: Layers,
  prompt_snapshot_file: FileCode2,
  prompt_block: ScrollText,
  gmas_context: Network,
  log: Terminal,
  log_block: Terminal,
  artifact_block: Archive,
  memory_path: FolderTree,
  harness_plan: GitBranch,
  harness_split: GitBranch,
  harness_selection: Trophy,
  harness_prune: Archive,
  harness_promotion: CheckSquare,
  harness_stage_result: CheckSquare,
  harness_result: Trophy,
};

// Tier → border color class for MemPalace nodes
const tierBorderClass = {
  always_on: 'border-purple-500/40',
  hot:       'border-blue-500/35',
  warm:      'border-sky-500/25',
  cold:      'border-zinc-600/30',
  transient: 'border-zinc-700/20',
};
const tierDotClass = {
  always_on: 'bg-purple-400',
  hot:       'bg-blue-400',
  warm:      'bg-sky-500',
  cold:      'bg-zinc-500',
  transient: 'bg-zinc-700',
};

const ALL_RUNS = '__all__';

function formatRunOptionLabel(run) {
  const created = run.created_at ? new Date(run.created_at).toLocaleString() : '';
  return [run.status, run.id, created].filter(Boolean).join(' · ');
}

function escapeOptionText(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

// Dagre layout helper
function getLayoutedElements(nodes, edges, direction = 'TB') {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({ rankdir: direction, ranksep: 140, nodesep: 60, edgesep: 30 });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: 220, height: 86 });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  const layoutedNodes = nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    return {
      ...node,
      position: {
        x: nodeWithPosition.x - 110,
        y: nodeWithPosition.y - 43,
      },
    };
  });

  return { nodes: layoutedNodes, edges };
}

// Custom node matching reference image
function MemoryNode({ data, selected }) {
  const Icon = nodeIcons[data.nodeType] || FileText;
  const timeAgo = getTimeAgo(data.createdAt);
  const linkCount = data.connectionCount || 0;
  const sourceLabel = data.source || data.scope || data.path || data.nodeType;
  const isSource = data.nodeType === 'source';
  const isPalace = typeof data.source === 'string' && data.source.startsWith('palace.');
  const tier = data.tier || '';
  const tierBorder = isPalace ? (tierBorderClass[tier] || 'border-sky-500/20') : '';
  const tierDot = isPalace ? (tierDotClass[tier] || 'bg-sky-500') : null;
  const iconClass = isSource ? 'text-emerald-300/80' : isPalace ? 'text-purple-400/80' : 'text-blue-400/70';

  return (
    <div
      className={`relative w-[220px] rounded-lg border px-4 py-3 cursor-pointer overflow-hidden transition-all duration-150 ${
        selected
          ? 'border-blue-500/60 bg-[#1a1f2e] shadow-[0_0_12px_rgba(59,130,246,0.15)]'
          : isSource
            ? 'border-emerald-500/25 bg-[#121a1a] hover:border-emerald-400/40 hover:bg-[#162121]'
            : isPalace
              ? `${tierBorder} bg-[#131520] hover:bg-[#18192a]`
              : 'border-[#2a3040] bg-[#141820] hover:border-[#3a4050] hover:bg-[#1a1f2e]'
      }`}
    >
      {/* Top handle with blue dot */}
      <Handle
        type="target"
        position={Position.Top}
        style={{
          background: '#4F8CFF',
          width: 8,
          height: 8,
          border: '2px solid #1a1f2e',
          top: -4,
        }}
      />
      {/* Bottom handle with blue dot */}
      <Handle
        type="source"
        position={Position.Bottom}
        style={{
          background: '#4F8CFF',
          width: 8,
          height: 8,
          border: '2px solid #1a1f2e',
          bottom: -4,
        }}
      />

      {/* Icon + Label */}
      <div className="flex items-center gap-2 mb-1.5">
        <Icon className={`h-3.5 w-3.5 shrink-0 ${iconClass}`} />
        <span className="text-[13px] font-medium text-foreground/90 truncate leading-tight">
          {data.label}
        </span>
      </div>

      <div className="mb-1.5 text-[11px] leading-tight text-muted-foreground/70 truncate">
        {sourceLabel}
      </div>

      {/* Metadata row */}
      <div className="flex items-center justify-between gap-2 text-[11px] text-muted-foreground/60">
        <div className="flex items-center gap-1.5">
          {tierDot && <div className={`w-1.5 h-1.5 rounded-full ${tierDot}`} title={`tier: ${tier}`} />}
          <span>{linkCount} links</span>
        </div>
        <span>{timeAgo}</span>
      </div>
    </div>
  );
}

const nodeTypes = { memory: MemoryNode };

function getTimeAgo(dateStr) {
  if (!dateStr) return '';
  try {
    const diff = Date.now() - new Date(dateStr).getTime();
    const seconds = Math.floor(diff / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  } catch {
    return '';
  }
}

export default function MemoryGraph() {
  const { activeWorkspace } = useWorkspace();
  const [rawNodes, setRawNodes] = useState([]);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [loading, setLoading] = useState(true);
  const [nodeCount, setNodeCount] = useState(0);
  const [edgeCount, setEdgeCount] = useState(0);
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState(ALL_RUNS);

  useEffect(() => {
    if (!activeWorkspace) return;
    listRuns({ workspace_id: activeWorkspace.id, limit: 100 })
      .then(data => {
        const nextRuns = data.runs || [];
        setRuns(nextRuns);
        setSelectedRunId(current => (
          current === ALL_RUNS || nextRuns.some(run => run.id === current)
            ? current
            : ALL_RUNS
        ));
      })
      .catch(() => {
        setRuns([]);
        setSelectedRunId(ALL_RUNS);
      });
  }, [activeWorkspace]);

  const fetchNodes = useCallback(async () => {
    if (!activeWorkspace) return;
    setLoading(true);
    try {
      const runFilter = selectedRunId === ALL_RUNS ? undefined : selectedRunId;
      const data = await listMemoryNodes(activeWorkspace.id, runFilter);
      setRawNodes(data);
      setNodeCount(data.length);

      // Convert to React Flow nodes
      const flowNodes = data.map((node) => ({
        id: node.id,
        type: 'memory',
        position: { x: 0, y: 0 },
        data: {
          label: node.label || node.title || node.id,
          nodeType: node.node_type || node.type || 'concept',
          content: node.content || node.summary || node.details || '',
          referenceCount: node.reference_count || 0,
          connectionCount: node.connections?.length || 0,
          createdAt: node.created_at,
          updatedAt: node.updated_at,
          source: node.source,
          path: node.path,
          scope: node.scope,
          tags: node.tags || [],
        },
      }));

      // Build edges
      const flowEdges = [];
      const edgeSet = new Set();
      let edgeIndex = 0;
      data.forEach(node => {
        (node.connections || []).forEach(targetId => {
          if (data.find(n => n.id === targetId)) {
            const edgeId = `${node.id}->${targetId}`;
            const reverseId = `${targetId}->${node.id}`;
            if (!edgeSet.has(edgeId) && !edgeSet.has(reverseId)) {
              edgeSet.add(edgeId);
              edgeIndex++;
              // Alternate between solid and dashed edges for visual variety
              const isDashed = edgeIndex % 3 === 0;
              flowEdges.push({
                id: edgeId,
                source: node.id,
                target: targetId,
                type: 'default',
                style: {
                  stroke: 'rgba(255,255,255,0.35)',
                  strokeWidth: 1.5,
                  strokeDasharray: isDashed ? '6 4' : 'none',
                },
              });
            }
          }
        });
      });

      setEdgeCount(flowEdges.length);

      // Apply dagre layout
      if (flowNodes.length > 0) {
        const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(flowNodes, flowEdges, 'TB');
        setNodes(layoutedNodes);
        setEdges(layoutedEdges);
      } else {
        setNodes([]);
        setEdges([]);
      }
    } catch (err) {
      console.error(err);
      toast.error('Failed to load memory graph');
    } finally {
      setLoading(false);
    }
  }, [activeWorkspace, selectedRunId, setNodes, setEdges]);

  useEffect(() => {
    fetchNodes();
  }, [fetchNodes]);

  const handleNodeClick = useCallback((event, node) => {
    const raw = rawNodes.find(n => n.id === node.id);
    setSelectedNode(raw);
  }, [rawNodes]);

  const handlePaneClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  const handleDelete = async (nodeId) => {
    try {
      const params = activeWorkspace?.id ? { workspace_id: activeWorkspace.id } : undefined;
      const result = await deleteMemoryNode(nodeId, params);
      if (result?.ok) {
        setSelectedNode(null);
        fetchNodes();
        const removedCount = result?.report?.removed_count ?? 0;
        toast.success(`Node deleted (${removedCount} артефактов)`);
      } else {
        const reason = result?.reason || 'node_type_not_deletable';
        toast.warning(
          reason === 'node_type_not_deletable'
            ? 'Этот тип узла нельзя удалить из графа (он построен из живых логов или системных файлов)'
            : `Не удалось удалить узел: ${reason}`
        );
      }
    } catch (err) {
      const detail = err?.response?.data?.reason || err?.message || 'Неизвестная ошибка';
      toast.error(`Failed to delete node: ${detail}`);
    }
  };

  if (!activeWorkspace) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-56px)]">
        <p className="text-sm text-muted-foreground">Select a workspace to view the memory graph</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-[calc(100vh-56px)]">
      {/* Page header */}
      <div className="px-6 py-4 border-b border-border">
        <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div>
            <h2 className="text-lg font-semibold tracking-tight">Memory Graph</h2>
            <p className="text-xs text-muted-foreground mt-0.5">
              {selectedRunId === ALL_RUNS
                ? `Workspace memory - ${nodeCount} nodes, ${edgeCount} edges`
                : `Run-scoped context - ${nodeCount} nodes, ${edgeCount} edges`}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-[10px] uppercase tracking-wider text-muted-foreground">Run</label>
            <select
              value={selectedRunId}
              onChange={(event) => {
                setSelectedRunId(event.target.value);
                setSelectedNode(null);
              }}
              className="h-8 max-w-[360px] rounded-md border border-border bg-secondary/30 px-2 text-xs text-foreground outline-none"
              data-testid="memory-run-filter"
            >
              <option value={ALL_RUNS}>All workspace memory</option>
              {runs.map(run => (
                <option
                  key={run.id}
                  value={run.id}
                  dangerouslySetInnerHTML={{ __html: escapeOptionText(formatRunOptionLabel(run)) }}
                />
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="flex flex-1 min-h-0 flex-col xl:flex-row">
        {/* Graph Canvas */}
        <div className="relative min-h-[420px] flex-1" data-testid="memory-graph-canvas">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : nodes.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center px-6">
              <div className="w-12 h-12 rounded-xl bg-secondary flex items-center justify-center mb-4">
                <GitBranch className="h-5 w-5 text-muted-foreground" />
              </div>
              <h3 className="text-sm font-semibold mb-1.5">No memory nodes yet</h3>
              <p className="text-xs text-muted-foreground max-w-sm leading-relaxed">
                {selectedRunId === ALL_RUNS
                  ? 'No workspace memory, logs, prompt snapshots, or runtime artifacts were found yet.'
                  : selectedRunId
                  ? 'No memory, logs, or result nodes were found for the selected run.'
                  : 'No memory nodes were found.'}
              </p>
            </div>
          ) : (
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={handleNodeClick}
              onPaneClick={handlePaneClick}
              nodeTypes={nodeTypes}
              fitView
              fitViewOptions={{ padding: 0.3 }}
              proOptions={{ hideAttribution: true }}
              defaultEdgeOptions={{
                type: 'default',
              }}
            >
              <Background variant="dots" gap={20} size={1} color="rgba(255,255,255,0.03)" />
              <Controls showInteractive={false} position="bottom-left" />
              <MiniMap
                nodeColor={() => '#4F8CFF'}
                maskColor="rgba(10,12,16,0.85)"
                style={{ background: '#0F1218', borderRadius: 8, border: '1px solid #1B2330' }}
              />
            </ReactFlow>
          )}

          {/* Refresh button */}
          <div className="absolute top-4 right-4 z-20">
            <Button
              variant="outline"
              size="sm"
              className="h-8 text-xs gap-1.5 bg-[#141820] border-[#2a3040] hover:bg-[#1a1f2e]"
              onClick={fetchNodes}
            >
              <RefreshCw className="h-3 w-3" /> Refresh
            </Button>
          </div>
        </div>

        {/* Node Details Panel (Right Side) */}
        <div
          data-testid="node-inspector"
          className="h-[260px] border-t border-border bg-card flex flex-col xl:h-auto xl:w-[320px] xl:border-l xl:border-t-0"
        >
          <div className="px-4 py-3 border-b border-border">
            <h3 className="text-sm font-semibold">Node Details</h3>
          </div>

          {selectedNode ? (
            <ScrollArea className="flex-1">
              <div className="p-4 space-y-4">
                {/* Node icon + title */}
                <div className="flex items-start gap-3">
                  <div className="flex h-8 w-8 items-center justify-center rounded-md bg-blue-500/10 border border-blue-500/20 shrink-0 mt-0.5">
                    {(() => {
                      const Icon = nodeIcons[selectedNode.node_type] || FileText;
                      return <Icon className="h-4 w-4 text-blue-400" />;
                    })()}
                  </div>
                  <div>
                    <h4 className="text-sm font-semibold leading-tight">{selectedNode.label}</h4>
                    <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
                      {selectedNode.node_type}
                    </span>
                  </div>
                </div>

                {/* Content */}
                {selectedNode.content && (
                  <div>
                    <label className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium mb-1 block">Content</label>
                    <p className="text-xs text-muted-foreground/80 leading-relaxed">{selectedNode.content}</p>
                  </div>
                )}

                {selectedNode.tags?.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {selectedNode.tags.slice(0, 10).map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-[10px] px-1.5 py-0">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                )}

                <Separator className="border-border/50" />

                {/* Connections */}
                <div>
                  <label className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium mb-2 block">
                    Connections ({selectedNode.connections?.length || 0})
                  </label>
                  <div className="space-y-1.5">
                    {selectedNode.connections?.length > 0 ? (
                      rawNodes
                        .filter(n => selectedNode.connections.includes(n.id))
                        .map(cn => {
                          const Icon = nodeIcons[cn.node_type] || FileText;
                          return (
                            <div key={cn.id} className="flex items-center gap-2 text-xs py-1.5 px-2.5 rounded-md bg-secondary/30 border border-border/30">
                              <ArrowRight className="h-3 w-3 text-muted-foreground/50 shrink-0" />
                              <Icon className="h-3 w-3 text-blue-400/60 shrink-0" />
                              <span className="truncate text-foreground/80">{cn.label}</span>
                            </div>
                          );
                        })
                    ) : (
                      <p className="text-xs text-muted-foreground/50">No outgoing connections</p>
                    )}
                  </div>
                </div>

                <Separator className="border-border/50" />

                {/* Metadata */}
                <div>
                  <label className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium mb-2 block">Metadata</label>
                  <div className="space-y-2 text-xs">
                    {selectedNode.source && (
                      <div className="flex justify-between gap-3">
                        <span className="text-muted-foreground/60">Source</span>
                        <span className="font-medium text-right truncate max-w-[190px]">{selectedNode.source}</span>
                      </div>
                    )}
                    {selectedNode.tier && (
                      <div className="flex justify-between gap-3">
                        <span className="text-muted-foreground/60">Tier</span>
                        <span className="font-medium text-right font-mono text-xs">{selectedNode.tier}</span>
                      </div>
                    )}
                    {selectedNode.verified !== undefined && (
                      <div className="flex justify-between gap-3">
                        <span className="text-muted-foreground/60">Verified</span>
                        <span className={`font-medium text-right text-xs ${selectedNode.verified ? 'text-emerald-400' : 'text-muted-foreground/50'}`}>
                          {selectedNode.verified ? 'yes' : 'no'}
                        </span>
                      </div>
                    )}
                    {selectedNode.scope && (
                      <div className="flex justify-between gap-3">
                        <span className="text-muted-foreground/60">Scope</span>
                        <span className="font-medium text-right truncate max-w-[190px]">{selectedNode.scope}</span>
                      </div>
                    )}
                    {selectedNode.path && (
                      <div className="flex justify-between gap-3">
                        <span className="text-muted-foreground/60">Path</span>
                        <span className="font-medium text-right truncate max-w-[190px]" title={selectedNode.path}>
                          {selectedNode.path}
                        </span>
                      </div>
                    )}
                    <div className="flex justify-between">
                      <span className="text-muted-foreground/60">References</span>
                      <span className="font-medium tabular-nums">{selectedNode.reference_count}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground/60">Links</span>
                      <span className="font-medium tabular-nums">{selectedNode.connections?.length || 0}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground/60">Created</span>
                      <span className="font-medium">{getTimeAgo(selectedNode.created_at)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground/60">Updated</span>
                      <span className="font-medium">{getTimeAgo(selectedNode.updated_at)}</span>
                    </div>
                  </div>
                </div>

                <Separator className="border-border/50" />

                <Button
                  variant="outline"
                  size="sm"
                  className="w-full gap-2 text-xs text-destructive/80 hover:text-destructive border-border hover:border-destructive/30 hover:bg-destructive/5"
                  onClick={() => handleDelete(selectedNode.id)}
                >
                  <Trash2 className="h-3 w-3" /> Delete Node
                </Button>
              </div>
            </ScrollArea>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-center px-6">
              <Network className="h-8 w-8 text-muted-foreground/30 mb-3" />
              <p className="text-sm text-muted-foreground/60 font-medium">Select a node to view details</p>
              <p className="text-[11px] text-muted-foreground/40 mt-1">Click any node in the graph</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
