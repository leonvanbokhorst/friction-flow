from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path
import json

@dataclass
class ExecutionResult:
    code: str
    output: Optional[str]
    timestamp: datetime
    status: str
    validation_results: Optional[Dict] = None
    metadata: Optional[Dict] = None

class ResultsStore:
    def __init__(self, storage_dir: str = "./data/results"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    async def store_execution(self, result: ExecutionResult) -> str:
        """Store execution result and return unique identifier"""
        result_id = f"{self.current_session}_{len(list(self.storage_dir.glob('*.json')))}"
        
        result_dict = {
            "code": result.code,
            "output": result.output,
            "timestamp": result.timestamp.isoformat(),
            "status": result.status,
            "validation_results": result.validation_results,
            "metadata": result.metadata
        }
        
        file_path = self.storage_dir / f"{result_id}.json"
        with open(file_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
            
        return result_id
        
    async def get_result(self, result_id: str) -> Optional[ExecutionResult]:
        """Retrieve stored execution result"""
        file_path = self.storage_dir / f"{result_id}.json"
        if not file_path.exists():
            return None
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            return ExecutionResult(
                code=data["code"],
                output=data["output"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                status=data["status"],
                validation_results=data["validation_results"],
                metadata=data["metadata"]
            )
            
    async def list_results(self, 
                          session_id: Optional[str] = None,
                          status: Optional[str] = None) -> List[str]:
        """List result IDs matching criteria"""
        results = []
        pattern = f"{session_id or '*'}_*.json"
        
        for file_path in self.storage_dir.glob(pattern):
            if status:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if data["status"] == status:
                        results.append(file_path.stem)
            else:
                results.append(file_path.stem)
                
        return results 