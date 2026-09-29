import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

class GeneticFeatureSelector:
    def __init__(self,population_size=50,crossover_rate=.8,mutation_rate=.1,max_generations=100,seed=1234):
        self.population_size=population_size; self.crossover_rate=crossover_rate; self.mutation_rate=mutation_rate; self.max_generations=max_generations; self.seed=seed
    def _fitness(self,X,y,mask):
        if mask.sum()==0: return 0.0
        clf=LogisticRegression(max_iter=1000,solver='liblinear',random_state=self.seed)
        cv=StratifiedKFold(3,shuffle=True,random_state=self.seed)
        return float(cross_val_score(clf,X[:,mask],y,cv=cv,scoring='accuracy').mean())
    def fit(self,X,y):
        rng=np.random.default_rng(self.seed); d=X.shape[1]
        pop=rng.integers(0,2,size=(self.population_size,d),dtype=np.int8); best=None
        for _ in range(self.max_generations):
            scores=np.array([self._fitness(X,y,m.astype(bool)) for m in pop]); order=np.argsort(scores)[::-1]
            if best is None or scores[order[0]]>best[0]: best=(scores[order[0]],pop[order[0]].copy())
            elite=pop[order[:max(2,self.population_size//5)]]; children=[]
            while len(children)<self.population_size:
                a,b=elite[rng.integers(len(elite))],elite[rng.integers(len(elite))]; child=a.copy()
                if rng.random()<self.crossover_rate:
                    point=rng.integers(1,d); child[point:]=b[point:]
                flips=rng.random(d)<self.mutation_rate; child[flips]^=1; children.append(child)
            pop=np.asarray(children,dtype=np.int8)
        self.best_score,self.mask=best; return self

class ParticleSwarmFeatureSelector:
    def __init__(self,swarm_size=50,c1=2.0,c2=2.0,w=.7,max_iterations=100,seed=1234):
        self.swarm_size=swarm_size; self.c1=c1; self.c2=c2; self.w=w; self.max_iterations=max_iterations; self.seed=seed
    def fit(self,X,y):
        rng=np.random.default_rng(self.seed); d=X.shape[1]; pos=rng.random((self.swarm_size,d)); vel=np.zeros_like(pos); pbest=pos.copy(); pscore=np.array([self._fitness(X,y,p) for p in pos]); gi=int(pscore.argmax()); gbest=pos[gi].copy(); gscore=pscore[gi]
        for _ in range(self.max_iterations):
            r1=rng.random(pos.shape); r2=rng.random(pos.shape); vel=self.w*vel+self.c1*r1*(pbest-pos)+self.c2*r2*(gbest-pos); pos=np.clip(pos+vel,0,1); score=np.array([self._fitness(X,y,p) for p in pos]); improved=score>pscore; pbest[improved]=pos[improved]; pscore[improved]=score[improved]; gi=int(pscore.argmax());
            if pscore[gi]>gscore: gbest=pbest[gi].copy(); gscore=pscore[gi]
        self.best_score=gscore; self.mask=gbest>=.5; return self
    def _fitness(self,X,y,p):
        mask=p>=.5
        if mask.sum()==0: return 0.0
        clf=LogisticRegression(max_iter=1000,solver='liblinear',random_state=self.seed); cv=StratifiedKFold(3,shuffle=True,random_state=self.seed)
        return float(cross_val_score(clf,X[:,mask],y,cv=cv,scoring='accuracy').mean())
