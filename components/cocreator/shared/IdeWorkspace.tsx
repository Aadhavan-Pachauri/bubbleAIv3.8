
import React, { useState, useEffect, useRef } from 'react';
import Split from 'react-split-grid';
import Editor from '@monaco-editor/react';
import { FileExplorer } from './FileExplorer';
import { Message, Project, Chat, WorkspaceMode, ChatWithProjectData } from '../../../types';
import { useCollaborativeCursors } from '../../../hooks/useCollaborativeCursors';
import { motion } from 'framer-motion';

export interface IdeWorkspaceProps {
  project: Project;
  chat: ChatWithProjectData | null;
  geminiApiKey: string;
  messages: Message[];
  isLoadingHistory: boolean;
  isCreatingChat: boolean;
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  onSendMessage: (text: string, files?: File[] | null, chat?: ChatWithProjectData | null, modelOverride?: string) => void;
  onChatUpdate: ((updates: Partial<Chat>) => void) | null;
  onActiveProjectUpdate: ((updates: Partial<Project>) => Promise<void>) | null;
  searchQuery: string;
  onSearchResultsChange: (indices: number[]) => void;
  currentSearchResultMessageIndex: number;
  isAdmin: boolean;
  workspaceMode: WorkspaceMode;
  projectType: 'website' | 'roblox_game';
  loadingMessage: string;
  onStop?: () => void;
}

const Cursor: React.FC<{ x: number; y: number; color: string; label: string }> = ({ x, y, color, label }) => (
    <motion.div
        className="absolute pointer-events-none z-50"
        initial={{ opacity: 0, scale: 0 }}
        animate={{ left: `${x}%`, top: `${y}%`, opacity: 1, scale: 1 }}
        transition={{ duration: 0.1, ease: "linear" }}
    >
        <svg
            className="w-5 h-5 drop-shadow-md"
            viewBox="0 0 24 24"
            fill={color}
            xmlns="http://www.w3.org/2000/svg"
            style={{ transform: 'rotate(-15deg) translate(-2px, -2px)' }}
        >
            <path d="M5.5 3.2L18.8 9.5L12.5 12.8L15.8 19.1L5.5 3.2Z" stroke="white" strokeWidth="1.5" />
        </svg>
        <div 
            className="absolute left-4 top-4 px-2 py-0.5 rounded text-xs font-bold text-white whitespace-nowrap shadow-sm"
            style={{ backgroundColor: color }}
        >
            {label}
        </div>
    </motion.div>
);

export const IdeWorkspace: React.FC<IdeWorkspaceProps> = (props) => {
    const { projectType, project, onActiveProjectUpdate } = props;
    
    // LIVE COLLAB HOOK
    const { cursors, containerRef } = useCollaborativeCursors(project.id);
    
    const [selectedFile, setSelectedFile] = useState<string | null>(null);
    const [editorContent, setEditorContent] = useState('');
    
    const prevFilesRef = useRef(Object.keys(project.files || {}));

    useEffect(() => {
        const currentFiles = Object.keys(project.files || {});
        const prevFiles = prevFilesRef.current;
        
        const newFile = currentFiles.find(f => !prevFiles.includes(f));
        if (newFile) {
            setSelectedFile(newFile);
        } else {
            if (selectedFile && !currentFiles.includes(selectedFile)) {
                setSelectedFile(null);
            }
            else if (!selectedFile && currentFiles.length > 0) {
                setSelectedFile(currentFiles[currentFiles.length - 1]);
            }
        }
        
        prevFilesRef.current = currentFiles;
    }, [project.files, selectedFile]);

    useEffect(() => {
        const content = selectedFile
            ? project.files?.[selectedFile]?.content ?? `// Error: Could not find content for ${selectedFile}`
            : `// Select a file to view its code`;
        setEditorContent(content);
    }, [selectedFile, project.files]);

    const handleFileSelect = (filePath: string) => {
        setSelectedFile(filePath);
    };

    const handleEditorChange = (value: string | undefined) => {
        if (selectedFile && value !== undefined && onActiveProjectUpdate) {
            const updatedFiles = {
                ...(project.files || {}),
                [selectedFile]: { content: value }
            };
            onActiveProjectUpdate({ files: updatedFiles });
        }
    };

    return (
        <div ref={containerRef} className="h-full w-full bg-bg-primary text-white relative">
            {/* Render Cursors Overlay */}
            {Object.entries(cursors).map(([id, cursor]: [string, any]) => (
                <Cursor key={id} x={cursor.x} y={cursor.y} color={cursor.color} label={cursor.user} />
            ))}

            <Split
                gridTemplateColumns="250px 8px 1fr"
                minSize={200}
                cursor="col-resize"
            >
                {(split: any) => (
                    <div className="grid h-full" {...split.getGridProps()}>
                        <div className="h-full bg-bg-secondary overflow-hidden">
                           <FileExplorer onFileSelect={handleFileSelect} project={project} />
                        </div>

                        <div className="h-full bg-bg-tertiary cursor-col-resize" {...split.getGutterProps('column', 1)} />
                        
                        <div className="h-full w-full overflow-hidden bg-[#1e1e1e]">
                            <Editor
                                height="100%"
                                path={selectedFile || 'default'} 
                                language={projectType === 'website' ? 'html' : 'lua'}
                                theme="vs-dark"
                                value={editorContent}
                                onChange={handleEditorChange}
                                options={{ 
                                    minimap: { enabled: false }, 
                                    fontSize: 14, 
                                    wordWrap: 'on',
                                    automaticLayout: true,
                                }}
                            />
                        </div>
                    </div>
                )}
            </Split>
        </div>
    );
};
