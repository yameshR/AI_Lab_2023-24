# Ex.No: 10  Logic Programming –  Simple queries from facts and rules
### DATE:29.04.2025                                                                            
### REGISTER NUMBER : 212222220059
### AIM: 
To write a prolog program to find the answer of query. 
###  Algorithm:
 Step 1: Start the program <br> 
 Step 2: Convert the sentence into First order Logic  <br> 
 Step 3:  Convert the sentence into Horn clause form  <br> 
 Step 4: Add rules and predicates in a program   <br> 
 Step 5:  Pass the query to program. <br> 
 Step 6: Prolog interpreter shows the output and return answer. <br> 
 Step 8:  Stop the program.
### Program:
### Task 1:
Construct the FOL representation for the following sentences <br> 
1.	John likes all kinds of food.  <br> 
2.	Apples are food.  <br> 
3.	Chicken is a food.  <br> 
4.	Sue eats everything Bill eats. <br> 
5.	 Bill eats peanuts  <br> 
   Convert into clause form and Prove that John like Apple by using Prolog. <br> 
### Program:
```
likes(john,X):-food(X).
eats(sue,X):-eats(bill,X).
eats(bill,peanuts).
food(peanuts).
food(apple).
food(chicken).
```
### Output:

![image](https://github.com/user-attachments/assets/4b71c9d8-1a6b-47ff-ad0f-298025b18d47)


### Task 2:
Consider the following facts and represent them in predicate form: <br>              
1.	Steve likes easy courses. <br> 
2.	Science courses are hard. <br> 
3. All the courses in Have fun department are easy <br> 
4. BK301 is Have fun department course.<br> 
Convert the facts in predicate form to clauses and then prove by resolution: “Steve likes BK301 course”<br> 

### Program:

```
easy(X) :- fun(X).
likes(steve, X) :- easy(X).
hard(X) :- science(X).
fun(bk301).
science(physics).
science(chemistry).
```


### Output:
![image](https://github.com/user-attachments/assets/722eee40-1567-415a-8a70-70ff154c0662)


### Task 3:
Consider the statement <br> 
“This is a crime for an American to sell weapons to hostile nations. The Nano , enemy of America has some missiles and its missiles were sold it by Colonal West who is an American” <br> 
Convert to Clause form and prove west is criminal by using Prolog.<br> 
### Program:

```
crime(X):-american(X),
          weapon(Y),
          hostile(Z),
           sells(X,Y,Z).
sells(west,Y,nano):-missile(Y),
    owns(nano,Y).
hostile(Z):-enemy(Z,D).
weapon(Y):-missile(Y).
american(west).
enemy(nano,american).
owns(nano,m1).
missile(m1).
```

### Output:

![image](https://github.com/user-attachments/assets/3434f545-76bc-4a01-9b7a-1d1c7964d8ff)


### Result:
Thus the prolog programs were executed successfully and the answer of query was found.
